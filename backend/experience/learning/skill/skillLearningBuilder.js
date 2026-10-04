/*
 * =========================================================
 * JESSICA EXPERIENCE SKILL LEARNING BUILDER
 * =========================================================
 *
 * Формирует канонические Learning Metrics
 * сохраняемого Experience Skill.
 *
 *
 * Канонический объект:
 *
 * skill.learning = {
 *     occurrences,
 *     successCount,
 *     failureCount,
 *     successRate,
 *     maturity,
 *     maturityLevel,
 *     confidence
 * }
 *
 *
 * Эти показатели относятся
 * к накопленному Experience,
 * а не только к одной версии Runtime.
 *
 * =========================================================
 */


import {
    isObject,
    normalizeText,
    normalizeUnit,
    normalizePositiveInteger,
    normalizeObjectArray
} from "./skillUtils.js";


/*
 * =========================================================
 * READ METRIC
 * =========================================================
 *
 * Новый контракт:
 *
 * proposedExperience.learning.*
 *
 * Fallback:
 *
 * proposedExperience.*
 *
 * нужен на время миграции.
 *
 * =========================================================
 */


function readMetric(
    proposedExperience,
    name
) {

    const learning =

        isObject(
            proposedExperience?.learning
        )

            ? proposedExperience.learning

            : {};


    if(
        learning[name] !== undefined
        &&
        learning[name] !== null
    ){

        return learning[name];

    }


    return proposedExperience?.[name];

}


/*
 * =========================================================
 * EXAMPLE STATISTICS
 * =========================================================
 */


function calculateExampleMetrics(
    proposedExperience
) {

    const examples =

        normalizeObjectArray(
            proposedExperience?.examples
        );


    const successCount =

        examples.filter(

            item =>
                item?.success === true

        )
        .length;


    const failureCount =

        examples.filter(

            item =>
                item?.success === false

        )
        .length;


    const successRate =

        examples.length > 0

            ?

            Number(

                (
                    successCount /
                    examples.length
                )
                .toFixed(2)

            )

            :

            0;


    return {

        examplesCount:
            examples.length,

        successCount,

        failureCount,

        successRate

    };

}


/*
 * =========================================================
 * CONFIDENCE
 * =========================================================
 *
 * Priority:
 *
 * explicit Builder confidence
 *      ↓
 * proposedExperience.learning.confidence
 *      ↓
 * proposedExperience.confidence
 *
 *
 * Нет искусственного default = 0.7.
 *
 * =========================================================
 */


function resolveConfidence(
    proposedExperience,
    confidence
) {

    const values = [

        confidence,

        readMetric(
            proposedExperience,
            "confidence"
        )

    ];


    for(
        const value
        of values
    ){

        if(
            value === undefined
            ||
            value === null
        ){

            continue;

        }


        const number =
            Number(value);


        if(
            Number.isFinite(
                number
            )
        ){

            return normalizeUnit(
                number
            );

        }

    }


    return 0;

}


/*
 * =========================================================
 * BUILD LEARNING
 * =========================================================
 */


export function buildSkillLearning({

    proposedExperience,

    confidence = null

}) {


    const calculated =

        calculateExampleMetrics(
            proposedExperience
        );


    const rawSuccessCount =

        readMetric(
            proposedExperience,
            "successCount"
        );


    const rawFailureCount =

        readMetric(
            proposedExperience,
            "failureCount"
        );


    const rawSuccessRate =

        readMetric(
            proposedExperience,
            "successRate"
        );


    const hasSuccessCount =

        Number.isFinite(
            Number(
                rawSuccessCount
            )
        );


    const hasFailureCount =

        Number.isFinite(
            Number(
                rawFailureCount
            )
        );


    const hasSuccessRate =

        Number.isFinite(
            Number(
                rawSuccessRate
            )
        );


    const successCount =

        hasSuccessCount

            ? Math.max(
                Number(
                    rawSuccessCount
                ),
                0
            )

            : calculated.successCount;


    const failureCount =

        hasFailureCount

            ? Math.max(
                Number(
                    rawFailureCount
                ),
                0
            )

            : calculated.failureCount;


    const occurrences =

        Math.max(

            normalizePositiveInteger(

                readMetric(
                    proposedExperience,
                    "occurrences"
                )

            ),

            calculated.examplesCount,

            1

        );


    const successRate =

        hasSuccessRate

            ? normalizeUnit(
                rawSuccessRate
            )

            : calculated.successRate;


    return {


        occurrences,


        successCount,


        failureCount,


        successRate,


        maturity:

            normalizeUnit(

                readMetric(
                    proposedExperience,
                    "maturity"
                )

            ),


        maturityLevel:

            normalizeText(

                readMetric(
                    proposedExperience,
                    "maturityLevel"
                )

            )

            ||

            null,


        confidence:

            resolveConfidence(

                proposedExperience,

                confidence

            )

    };

}


/*
 * =========================================================
 * RUNTIME STATISTICS
 * =========================================================
 *
 * statistics != learning
 *
 *
 * learning:
 *
 * накопленное доказательство качества
 * всей Experience lineage.
 *
 *
 * statistics:
 *
 * фактическое использование
 * конкретной активной версии Skill.
 *
 *
 * Поэтому новая версия Skill
 * начинает Runtime Statistics с нуля.
 *
 * История обучения при этом НЕ теряется,
 * потому что находится в skill.learning.
 *
 * =========================================================
 */


export function buildInitialRuntimeStatistics()
{

    return {


        successfulRuns:
            0,


        failedRuns:
            0,


        lastUsedAt:
            null,


        lastResult:
            null

    };

}
