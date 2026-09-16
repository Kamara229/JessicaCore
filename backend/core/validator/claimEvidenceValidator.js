import {
    executeAIWithRetry
} from "../../ai/aiRetry.js";

import {
    validatorChat
} from "../../ai/validatorClient.js";

import {
    buildClaimEvidenceInstructions
} from "./claim/claimEvidencePrompt.js";


/*
 * =========================================================
 * JESSICA CLAIM EVIDENCE VALIDATOR
 * =========================================================
 *
 * Проверяет существенные фактические утверждения
 * итогового ответа по реально загруженным источникам.
 *
 *
 * Ответственность:
 *
 * Answer
 *    +
 * Source Content
 *          ↓
 * Claim Validation
 *          ↓
 * подтверждён / не подтверждён
 *
 *
 * НЕ отвечает за:
 *
 * - создание AI клиента;
 * - выбор модели;
 * - API ключи;
 * - выполнение инструментов;
 * - поиск источников.
 *
 *
 * AI:
 *
 * validatorClient.js
 *
 * Retry:
 *
 * ai/aiRetry.js
 *
 * =========================================================
 */


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
        String(
            text || ""
        )
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
        value.indexOf(
            "{"
        );


    const lastBrace =
        value.lastIndexOf(
            "}"
        );


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


    return value.trim();

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


    fetchResults.forEach(
        (
            result,
            index
        ) => {


            parts.push(

                [

                    `SOURCE ${index + 1}`,

                    `URL: ${
                        String(
                            result?.data?.url || ""
                        )
                    }`,


                    `TITLE: ${
                        String(
                            result?.data?.title || ""
                        )
                    }`,


                    "",


                    String(
                        result?.data?.content || ""
                    )

                ]

                .join(
                    "\n"
                )

            );

        }

    );


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


    return {

        claim:
            text,


        supported:
            claim.supported === true,


        evidence:
            typeof claim.evidence === "string"
                ? claim.evidence
                    .trim()
                    .slice(
                        0,
                        MAX_EVIDENCE_LENGTH
                    )
                : "",


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


    if (
        !claim.evidence
    ) {

        return {

            ...claim,

            supported:
                false,

            reason:
                claim.reason ||
                "Не указан подтверждающий фрагмент"

        };

    }


    const source =
        sourceText
            .toLowerCase()
            .replace(
                /\s+/g,
                " "
            )
            .trim();


    const evidence =
        claim.evidence
            .toLowerCase()
            .replace(
                /\s+/g,
                " "
            )
            .trim();



    if (
        !source.includes(
            evidence
        )
    ) {

        return {

            ...claim,

            supported:
                false,

            reason:
                "Evidence отсутствует в загруженном источнике"

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


            return await validatorChat(

                [

                    {

                        role:
                            "system",

                        content:
                            instructions

                    },


                    {

                        role:
                            "user",

                        content:

                            [

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

                            ]

                            .join(
                                "\n"
                            )

                    }

                ]

            );


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
                "Проверка источника не требуется"

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

            success:
                true,

            valid:
                false,

            shouldRetry:
                true,

            claims:
                [],

            reason:
                "Нет загруженного источника"

        };

    }



    const answer =
        String(
            answerResult?.text || ""
        )
        .trim();



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
                "Ответ отсутствует"

        };

    }



    const sourceText =
        buildSourceText(
            fetchResults
        );



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
                    "Validator вернул пустой ответ"

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
                    "Validator вернул некорректный JSON"

            };

        }



        const claims =
            Array.isArray(
                parsed?.claims
            )

                ? parsed.claims

                    .map(
                        normalizeClaim
                    )

                    .filter(
                        Boolean
                    )

                    .map(
                        claim =>
                            verifyEvidenceText(
                                claim,
                                sourceText
                            )
                    )

                : [];



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

                claims,

                reason:
                    "Не найдены проверяемые утверждения"

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

                success:
                    true,

                valid:
                    false,

                shouldRetry:
                    true,

                claims,

                reason:
                    `Не подтверждено утверждений: ${unsupported.length}`

            };

        }



        return {

            success:
                true,

            valid:
                true,

            shouldRetry:
                false,

            claims,

            reason:
                "Все утверждения подтверждены источником"

        };


    } catch (error) {


        console.error(
            "Claim Evidence Validator error:",
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
