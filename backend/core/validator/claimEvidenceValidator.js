import { buildComparisonClaimInstructions } from "./claim/comparisonClaimPrompt.js";
import { usePairedComparisonContext } from "../composer/comparisonEvidencePolicy.js";
import { buildOriginalSourceEvidenceContext } from "./claim/claimEvidenceMatcher.js";
import { checkPairedClaimCoverage } from "./claim/pairedClaimCoverage.js";
import { partitionComparisonClaims } from "./claim/claimUncertaintyScope.js";
import { validateComparisonCells } from "./claim/comparisonCellValidator.js";
import {
    executeAIWithRetry
} from "../../ai/aiRetry.js";


import {
    validatorChat
} from "../../ai/validatorClient.js";


import {
    buildClaimEvidenceInstructions
} from "./claim/claimEvidencePrompt.js";


import {
    buildClaimEvidenceContext
} from "./claim/claimEvidenceContext.js";


import {
    parseClaimEvidenceResponse
} from "./claim/claimEvidenceResponse.js";


/*
 * =========================================================
 * JESSICA CLAIM EVIDENCE VALIDATOR v2
 * =========================================================
 *
 * Проверяет существенные утверждения
 * итогового ответа по реально
 * загруженным источникам.
 *
 *
 * Flow:
 *
 * Task
 * +
 * Answer
 * +
 * web_fetch Sources
 *        ↓
 * Claim Evidence Context
 *        ↓
 * deterministic relevance selection
 *        ↓
 * compact source snippets
 *        ↓
 * Claim AI Validation
 *        ↓
 * local evidence verification
 *
 *
 * Важно:
 *
 * Полные web_fetch documents
 * больше НЕ отправляются AI.
 *
 * Это защищает:
 *
 * - от Groq ITPM overflow;
 * - от лишних токенов;
 * - от низкого signal/noise;
 * - от больших HTML/text payload.
 *
 * =========================================================
 */


/*
 * =========================================================
 * AI REQUEST
 * =========================================================
 */


async function requestClaimValidation(

    task,

    answer,

    sourceText,
    comparison=false

) {

    const instructions =

        (comparison ? buildComparisonClaimInstructions() : buildClaimEvidenceInstructions());


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

                                String(
                                    answer || ""
                                ),


                                "",


                                "РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ РЕАЛЬНО ЗАГРУЖЕННЫХ ИСТОЧНИКОВ:",

                                sourceText

                            ]
                                .join(
                                    "\n"
                                )

                    }

                ],
                {comparison}

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
 * UNAVAILABLE
 * =========================================================
 */


function buildUnavailableResult(
    reason,
    status = 0
) {

    return {

        success:
            false,

        unavailable:
            true,

        valid:
            false,

        shouldRetry:
            false,

        claims:
            [],

        status,

        reason:

            reason

            ||

            "Claim Evidence Validator unavailable"

    };

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
     * MODE
     * =====================================================
     */


    if(
        plan?.evidence?.mode !==
        "source_content"
    ){

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


    /*
     * =====================================================
     * ANSWER
     * =====================================================
     */


    const answer =

        String(
            answerResult?.text || ""
        )
            .trim();


    if(
        !answer
    ){

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


    // The main comparison validator checks every asserted source cell;
    // unchecked table rows must never become a successful Experience.
    if (usePairedComparisonContext(task,plan,taskRunResult) &&
        /таблиц|table/iu.test(String(task||""))) {
        return await validateComparisonCells(task,taskRunResult,answer);
    }

    /*
     * =====================================================
     * COMPACT SOURCE CONTEXT
     * =====================================================
     */


    const context =

        buildClaimEvidenceContext({

            task,

            answer,

            taskRunResult

        });


    console.log(

        "Jessica Validator claim context:",

        JSON.stringify({

            sourceCount:
                context.sourceCount,

            chunkCount:
                context.chunkCount,

            sourceChars:
                context.sourceChars,

            queryTerms:
                context.queryTerms.length

        })

    );


    /*
     * Нет web_fetch source.
     */


    if(
        context.sourceCount === 0
    ){

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


    if(
        !context.sourceText
    ){

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
                "Не удалось сформировать Source Context"

        };

    }


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

                context.sourceText,
                usePairedComparisonContext(task,plan,taskRunResult)

            );


        const raw =

            response
                ?.choices
                ?.[0]
                ?.message
                ?.content;


        /*
         * =================================================
         * PARSE
         * =================================================
         */


        const parsed =

            parseClaimEvidenceResponse(

                raw,

                context.sourceText,

                buildOriginalSourceEvidenceContext(taskRunResult)

            );


        /*
         * AI технически ответил,
         * но ответ невозможно использовать.
         *
         * Это считается unavailable,
         * чтобы Validator Coordinator
         * мог включить semantic fallback.
         */


        if(
            parsed.success !== true
        ){

            return buildUnavailableResult(

                parsed.error

                ||

                "Некорректный ответ Claim Evidence Validator"

            );

        }


        const isComparison = usePairedComparisonContext(task,plan,taskRunResult);
        const sourceUrls=(taskRunResult?.results||[])
            .filter(item=>item?.tool==="web_fetch"&&item?.success===true)
            .map(item=>String(item?.data?.url||""));
        const normalized=isComparison
            ? partitionComparisonClaims(parsed.claims,answer,sourceUrls)
            : {claims:parsed.claims,ignoredUncertainty:0};
        const claims=normalized.claims;
        if(normalized.ignoredUncertainty) {
            console.log("Jessica Uncertainty Claim Filter:", {
                skipped:normalized.ignoredUncertainty,
                kept:claims.length
            });
        }


        /*
         * =================================================
         * NO CLAIMS
         * =================================================
         */


        if(
            claims.length === 0
        ){

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


        if(
            unsupported.length > 0
        ){

            return {

                success:
                    true,

                valid:
                    false,

                shouldRetry:
                    true,

                claims,

                reason:

                    `Не подтверждено утверждений: ${unsupported.length}. Ошибочные утверждения: ${unsupported.slice(0, 4).map(x => JSON.stringify(x.claim) + (x.reason ? " (" + x.reason + ")" : "")).join("; ")}. При повторной генерации исключи эти тезисы либо прямо укажи, что условие не подтверждено; не делай вывод об отсутствии по молчанию источника.`

            };

        }


        const coverage = checkPairedClaimCoverage({
            task, plan, taskRunResult, answer, claims
        });
        if (!coverage.valid) {
            return { success:true, valid:false, shouldRetry:false,
                claims, reason:coverage.reason };
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
                "Все утверждения подтверждены источником"

        };


    }catch(error){


        console.error(

            "Claim Evidence Validator error:",

            error

        );


        return buildUnavailableResult(

            error?.message

            ||

            "Ошибка Claim Evidence Validator",

            error?.status || 0

        );

    }

}
