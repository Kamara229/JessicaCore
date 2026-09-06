import OpenAI from "openai";


/*
 * =========================================================
 * JESSICA CLAIM EVIDENCE VALIDATOR
 * =========================================================
 *
 * Проверяет конкретные проверяемые утверждения
 * итогового ответа по реально загруженному источнику.
 *
 * Для каждого существенного утверждения Validator
 * должен указать подтверждающий фрагмент evidence.
 *
 * Если подтверждения нет:
 * SUPPORTED=false
 */


const groq =
    process.env.GROQ_API_KEY
        ? new OpenAI({
            apiKey: process.env.GROQ_API_KEY,
            baseURL: "https://api.groq.com/openai/v1"
        })
        : null;


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const MAX_CONTENT_LENGTH = 20000;

const MAX_EVIDENCE_LENGTH = 500;


/*
 * =========================================================
 * CLEAN JSON
 * =========================================================
 */


function cleanJsonText(text) {

    let value =
        String(text || "")
            .replace(/```json/gi, "")
            .replace(/```/g, "")
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

    const parts = [];


    for (
        let index = 0;
        index < fetchResults.length;
        index++
    ) {

        const result =
            fetchResults[index];


        const url =
            String(
                result.data?.url || ""
            ).trim();


        const title =
            String(
                result.data?.title || ""
            ).trim();


        const content =
            String(
                result.data?.content || ""
            ).trim();


        parts.push(
            [
                `SOURCE ${index + 1}`,
                `URL: ${url}`,
                `TITLE: ${title}`,
                "",
                content
            ].join("\n")
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
 * AI не может просто придумать evidence.
 *
 * Проверяем, что указанный фрагмент действительно
 * присутствует в реальном содержимом fetch.
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

            supported: false,

            reason:
                claim.reason ||
                "Validator не указал подтверждающий фрагмент"
        };
    }


    const normalizedSource =
        sourceText
            .toLowerCase()
            .replace(/\s+/g, " ")
            .trim();


    const normalizedEvidence =
        claim.evidence
            .toLowerCase()
            .replace(/\s+/g, " ")
            .trim();


    if (
        !normalizedSource.includes(
            normalizedEvidence
        )
    ) {

        return {
            ...claim,

            supported: false,

            reason:
                "Указанный evidence отсутствует в реально загруженном источнике"
        };
    }


    return claim;
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
     * Для задач без source_content
     * эта проверка пока не требуется.
     */
    if (
        plan?.evidence?.mode !==
        "source_content"
    ) {

        return {
            success: true,

            valid: true,

            shouldRetry: false,

            claims: [],

            reason:
                "Проверка утверждений по источнику не требуется"
        };
    }


    const fetchResults =
        findFetchResults(
            taskRunResult
        );


    if (
        fetchResults.length === 0
    ) {

        return {
            success: true,

            valid: false,

            shouldRetry: true,

            claims: [],

            reason:
                "Нет загруженного источника для проверки утверждений"
        };
    }


    const answer =
        String(
            answerResult?.text || ""
        ).trim();


    if (!answer) {

        return {
            success: true,

            valid: false,

            shouldRetry: true,

            claims: [],

            reason:
                "Итоговый ответ отсутствует"
        };
    }


    /*
     * Если AI недоступен,
     * не можем выполнить claim-level проверку.
     */
    if (!groq) {

        return {
            success: false,

            unavailable: true,

            valid: true,

            shouldRetry: false,

            claims: [],

            reason:
                "Claim Evidence Validator недоступен"
        };
    }


    const sourceText =
        buildSourceText(
            fetchResults
        );


    try {

        const response =
            await groq.chat.completions.create({

                model:
                    "openai/gpt-oss-20b",

                temperature:
                    0,

                messages: [
                    {
                        role: "system",

                        content: [
                            "Ты Claim Evidence Validator системы Jessica Core.",
                            "",
                            "Ты НЕ отвечаешь пользователю.",
                            "Ты проверяешь итоговый ответ только по переданным источникам.",
                            "",
                            "Выдели только существенные фактические утверждения,",
                            "которые важны для выполнения задачи.",
                            "",
                            "Для каждого утверждения укажи:",
                            "- claim",
                            "- supported",
                            "- evidence",
                            "- sourceUrl",
                            "- reason",
                            "",
                            "supported=true разрешено ставить ТОЛЬКО если",
                            "в переданном содержимом есть конкретный текстовый фрагмент,",
                            "который подтверждает это утверждение.",
                            "",
                            "Поле evidence должно содержать короткий дословный фрагмент",
                            "из переданного источника.",
                            "",
                            "Не пересказывай evidence своими словами.",
                            "Не придумывай evidence.",
                            "",
                            "Если нужного подтверждения нет,",
                            "supported=false и evidence=\"\".",
                            "",
                            "Не считай утверждение подтверждённым только потому,",
                            "что оно выглядит правдоподобно.",
                            "",
                            "Верни только JSON:",
                            JSON.stringify({
                                claims: [
                                    {
                                        claim:
                                            "проверяемое утверждение",

                                        supported:
                                            true,

                                        evidence:
                                            "дословный фрагмент источника",

                                        sourceUrl:
                                            "URL источника",

                                        reason:
                                            "краткая причина"
                                    }
                                ]
                            })
                        ].join("\n")
                    },

                    {
                        role: "user",

                        content: [
                            "ИСХОДНАЯ ЗАДАЧА:",
                            String(task || ""),

                            "",
                            "ИТОГОВЫЙ ОТВЕТ:",
                            answer,

                            "",
                            "РЕАЛЬНО ЗАГРУЖЕННЫЕ ИСТОЧНИКИ:",
                            sourceText
                        ].join("\n")
                    }
                ]

            });


        const raw =
            response
                ?.choices
                ?.[0]
                ?.message
                ?.content;


        if (!raw) {

            return {
                success: false,

                valid: true,

                shouldRetry: false,

                claims: [],

                reason:
                    "Claim Evidence Validator вернул пустой ответ"
            };
        }


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
                success: false,

                valid: true,

                shouldRetry: false,

                claims: [],

                reason:
                    "Claim Evidence Validator вернул некорректный JSON"
            };
        }


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
         * Если Validator вообще не выделил
         * утверждений из фактического ответа,
         * это нельзя считать полноценной проверкой.
         */
        if (
            claims.length === 0
        ) {

            return {
                success: true,

                valid: false,

                shouldRetry: true,

                claims: [],

                reason:
                    "Не удалось выделить проверяемые утверждения итогового ответа"
            };
        }


        const unsupported =
            claims.filter(
                claim =>
                    claim.supported !== true
            );


        if (
            unsupported.length > 0
        ) {

            return {
                success: true,

                valid: false,

                shouldRetry: true,

                claims,

                reason:
                    (
                        `Не подтверждено существенных утверждений: ` +
                        `${unsupported.length}`
                    )
            };
        }


        return {
            success: true,

            valid: true,

            shouldRetry: false,

            claims,

            reason:
                "Все существенные утверждения подтверждены реальным содержимым источника"
        };


    } catch (error) {

        console.error(
            "Claim Evidence Validator error:",
            error
        );


        return {
            success: false,

            unavailable: true,

            valid: true,

            shouldRetry: false,

            claims: [],

            reason:
                error?.message ||
                "Ошибка Claim Evidence Validator"
        };
    }

}
