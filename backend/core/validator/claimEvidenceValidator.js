import OpenAI from "openai";

import {
    executeAIWithRetry
} from "../../ai/aiRetry.js";

import {
    buildClaimEvidenceInstructions
} from "./claim/claimEvidencePrompt.js";


/*
 * =========================================================
 * JESSICA CLAIM EVIDENCE VALIDATOR
 * =========================================================
 *
 * Проверяет только существенные фактические
 * утверждения итогового ответа.
 *
 * Подробная политика выделения claims:
 *
 * validator/claim/claimEvidencePrompt.js
 *
 * Технический retry AI:
 *
 * ai/aiRetry.js
 */


/*
 * =========================================================
 * AI CLIENT
 * =========================================================
 */


const groq =
    process.env.GROQ_API_KEY
        ? new OpenAI({
            apiKey:
                process.env.GROQ_API_KEY,

            baseURL:
                "https://api.groq.com/openai/v1"
        })
        : null;


const CLAIM_VALIDATOR_MODEL =
    "openai/gpt-oss-20b";


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const MAX_CONTENT_LENGTH =
    20000;


const MAX_EVIDENCE_LENGTH =
    500;


/*
 * =========================================================
 * CLEAN JSON
 * =========================================================
 */


function cleanJsonText(
    text
) {

    let value =
        String(text || "")
            .replace(
                /```json/gi,
                ""
            )
            .replace(
                /```/g,
                ""
            )
            .trim();


    const firstBrace =
        value.indexOf("{");


    const lastBrace =
        value.lastIndexOf("}");


    if (
        firstBrace !== -1 &&
        lastBrace > firstBrace
    ) {

        value =
            value.slice(
                firstBrace,
                lastBrace + 1
            );

    }


    return value;

}


/*
 * =========================================================
 * FIND FETCH RESULTS
 * =========================================================
 */


function findFetchResults(
    taskRunResult
) {

    const results =
        Array.isArray(
            taskRunResult?.results
        )
            ? taskRunResult.results
            : [];


    return results.filter(
        result =>
            result?.tool === "web_fetch" &&
            result?.success === true &&
            typeof result?.data?.content === "string" &&
            result.data.content.trim()
    );

}


/*
 * =========================================================
 * BUILD SOURCE TEXT
 * =========================================================
 */


function buildSourceText(
    fetchResults
) {

    const parts =
        [];


    for (
        let index = 0;
        index < fetchResults.length;
        index++
    ) {

        const result =
            fetchResults[index];


        const url =
            String(
                result?.data?.url || ""
            ).trim();


        const title =
            String(
                result?.data?.title || ""
            ).trim();


        const content =
            String(
                result?.data?.content || ""
            ).trim();


        parts.push(
            [
                `SOURCE ${index + 1}`,
                `URL: ${url}`,
                `TITLE: ${title}`,
                "",
                content
            ].join(
                "\n"
            )
        );

    }


    return parts
        .join(
            "\n\n====================\n\n"
        )
        .slice(
            0,
            MAX_CONTENT_LENGTH
        );

}


/*
 * =========================================================
 * NORMALIZE CLAIM
 * =========================================================
 */


function normalizeClaim(
    claim
) {

    if (
        !claim ||
        typeof claim !== "object"
    ) {

        return null;

    }


    const text =
        typeof claim.claim === "string"
            ? claim.claim.trim()
            : "";


    if (!text) {

        return null;

    }


    const evidence =
        typeof claim.evidence === "string"
            ? claim.evidence
                .trim()
                .slice(
                    0,
                    MAX_EVIDENCE_LENGTH
                )
            : "";


    return {
        claim:
            text,

        supported:
            claim.supported === true,

        evidence,

        sourceUrl:
            typeof claim.sourceUrl === "string"
                ? claim.sourceUrl.trim()
                : "",

        reason:
            typeof claim.reason === "string"
                ? claim.reason.trim()
                : ""
    };

}


/*
 * =========================================================
 * VERIFY EVIDENCE TEXT
 * =========================================================
 *
 * Даже если AI говорит supported=true,
 * код сам проверяет, что evidence реально
 * присутствует в загруженном содержимом.
 */


function verifyEvidenceText(
    claim,
    sourceText
) {

    if (
        claim.supported !== true
    ) {

        return claim;

    }


    if (!claim.evidence) {

        return {
            ...claim,

            supported:
                false,

            reason:
                claim.reason ||
                "Validator не указал подтверждающий фрагмент"
        };

    }


    const normalizedSource =
        sourceText
            .toLowerCase()
            .replace(
                /\s+/g,
                " "
            )
            .trim();


    const normalizedEvidence =
        claim.evidence
            .toLowerCase()
            .replace(
                /\s+/g,
                " "
            )
            .trim();


    if (
        !normalizedSource.includes(
            normalizedEvidence
        )
    ) {

        return {
            ...claim,

            supported:
                false,

            reason:
                "Указанный evidence отсутствует в реально загруженном источнике"
        };

    }


    return claim;

}


/*
 * =========================================================
 * AI REQUEST
 * =========================================================
 */


async function requestClaimValidation(
    task,
    answer,
    sourceText
) {

    const instructions =
        buildClaimEvidenceInstructions();


    return await executeAIWithRetry(
        async () => {

            return await groq
                .chat
                .completions
                .create({

                    model:
                        CLAIM_VALIDATOR_MODEL,

                    temperature:
                        0,

                    messages: [
                        {
                            role:
                                "system",

                            content:
                                instructions
                        },

                        {
                            role:
                                "user",

                            content: [
                                "ИСХОДНАЯ ЗАДАЧА:",
                                String(
                                    task || ""
                                ),

                                "",
                                "ИТОГОВЫЙ ОТВЕТ:",
                                answer,

                                "",
                                "РЕАЛЬНО ЗАГРУЖЕННЫЕ ИСТОЧНИКИ:",
                                sourceText
                            ].join(
                                "\n"
                            )
                        }
                    ]

                });

        },
        {
            label:
                "Claim Evidence Validator"
        }
    );

}


/*
 * =========================================================
 * PUBLIC VALIDATOR
 * =========================================================
 */


export async function validateClaimEvidence(
    task,
    plan,
    taskRunResult,
    answerResult
) {

    /*
     * =====================================================
     * SOURCE CONTENT NOT REQUIRED
     * =====================================================
     */


    if (
        plan?.evidence?.mode !==
        "source_content"
    ) {

        return {
            success:
                true,

            valid:
                true,

            shouldRetry:
                false,

            claims:
                [],

            reason:
                "Проверка утверждений по источнику не требуется"
        };

    }


    /*
     * =====================================================
     * FETCH RESULTS
     * =====================================================
     */


    const fetchResults =
        findFetchResults(
            taskRunResult
        );


    if (
        fetchResults.length === 0
    ) {

        return {
            success:
                true,

            valid:
                false,

            shouldRetry:
                true,

            claims:
                [],

            reason:
                "Нет загруженного источника для проверки утверждений"
        };

    }


    /*
     * =====================================================
     * ANSWER
     * =====================================================
     */


    const answer =
        String(
            answerResult?.text || ""
        ).trim();


    if (!answer) {

        return {
            success:
                true,

            valid:
                false,

            shouldRetry:
                true,

            claims:
                [],

            reason:
                "Итоговый ответ отсутствует"
        };

    }


    /*
     * =====================================================
     * AI UNAVAILABLE
     * =====================================================
     */


    if (!groq) {

        return {
            success:
                false,

            unavailable:
                true,

            valid:
                true,

            shouldRetry:
                false,

            claims:
                [],

            reason:
                "Claim Evidence Validator недоступен"
        };

    }


    /*
     * =====================================================
     * SOURCE CONTEXT
     * =====================================================
     */


    const sourceText =
        buildSourceText(
            fetchResults
        );


    /*
     * =====================================================
     * AI VALIDATION
     * =====================================================
     */


    try {

        const response =
            await requestClaimValidation(
                task,
                answer,
                sourceText
            );


        const raw =
            response
                ?.choices
                ?.[0]
                ?.message
                ?.content;


        /*
         * =================================================
         * EMPTY RESPONSE
         * =================================================
         */


        if (!raw) {

            return {
                success:
                    false,

                valid:
                    true,

                shouldRetry:
                    false,

                claims:
                    [],

                reason:
                    "Claim Evidence Validator вернул пустой ответ"
            };

        }


        /*
         * =================================================
         * PARSE JSON
         * =================================================
         */


        let parsed;


        try {

            parsed =
                JSON.parse(
                    cleanJsonText(
                        raw
                    )
                );

        } catch {

            console.error(
                "Claim Evidence Validator invalid JSON:",
                raw
            );


            return {
                success:
                    false,

                valid:
                    true,

                shouldRetry:
                    false,

                claims:
                    [],

                reason:
                    "Claim Evidence Validator вернул некорректный JSON"
            };

        }


        /*
         * =================================================
         * NORMALIZE CLAIMS
         * =================================================
         */


        const rawClaims =
            Array.isArray(
                parsed?.claims
            )
                ? parsed.claims
                : [];


        const claims =
            rawClaims
                .map(
                    normalizeClaim
                )
                .filter(Boolean)
                .map(
                    claim =>
                        verifyEvidenceText(
                            claim,
                            sourceText
                        )
                );


        /*
         * =================================================
         * NO CLAIMS
         * =================================================
         *
         * Для source_content задачи должен существовать
         * хотя бы один существенный проверяемый факт.
         */


        if (
            claims.length === 0
        ) {

            return {
                success:
                    true,

                valid:
                    false,

                shouldRetry:
                    true,

                claims:
                    [],

                reason:
                    "Не удалось выделить существенные проверяемые утверждения итогового ответа"
            };

        }


        /*
         * =================================================
         * UNSUPPORTED CLAIMS
         * =================================================
         */


        const unsupported =
            claims.filter(
                claim =>
                    claim.supported !== true
            );


        if (
            unsupported.length > 0
        ) {

            return {
                success:
                    true,

                valid:
                    false,

                shouldRetry:
                    true,

                claims,

                reason:
                    (
                        `Не подтверждено существенных утверждений: ` +
                        `${unsupported.length}`
                    )
            };

        }


        /*
         * =================================================
         * SUCCESS
         * =================================================
         */


        return {
            success:
                true,

            valid:
                true,

            shouldRetry:
                false,

            claims,

            reason:
                "Все существенные утверждения подтверждены реальным содержимым источника"
        };


    } catch (error) {

        /*
         * Сюда попадём после исчерпания
         * технических попыток aiRetry.js
         * либо при неретраебельной ошибке.
         */


        console.error(
            "Claim Evidence Validator final error:",
            error
        );


        return {
            success:
                false,

            unavailable:
                true,

            valid:
                true,

            shouldRetry:
                false,

            claims:
                [],

            status:
                error?.status || 0,

            reason:
                error?.message ||
                "Ошибка Claim Evidence Validator"
        };

    }

}
