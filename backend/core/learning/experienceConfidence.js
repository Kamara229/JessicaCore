/*
 * =========================================================
 * JESSICA EXPERIENCE CONFIDENCE v2
 * =========================================================
 *
 * Расчёт evidence-метрик Experience.
 *
 *
 * Flow:
 *
 * Execution Trace
 *        ↓
 * Examples
 *        ↓
 * Occurrences
 *        ↓
 * Success Rate
 *        ↓
 * Confidence
 *        ↓
 * Maturity
 *
 *
 * Ответственность:
 *
 * - определить количество наблюдений;
 * - нормализовать Examples;
 * - определить Success Rate;
 * - рассчитать Confidence;
 * - рассчитать числовую Maturity.
 *
 *
 * НЕ:
 *
 * - ищет Experience Pattern;
 * - создаёт Candidate;
 * - создаёт Skill;
 * - принимает Learning Decision;
 * - сохраняет Experience;
 * - работает с Supabase.
 *
 *
 * ВАЖНО:
 *
 * maturity всегда число:
 *
 * 0.0 ... 1.0
 *
 * Текстовые состояния:
 *
 * NEW
 * LEARNING
 * READY
 *
 * являются только отображением уровня зрелости
 * и не используются как числовая метрика.
 *
 * =========================================================
 */





/*
 * =========================================================
 * NORMALIZE NUMBER
 * =========================================================
 */


function normalizeNumber(
    value
) {


    const number =
        Number(value);



    return Number.isFinite(number)

        ? number

        : 0;

}





/*
 * =========================================================
 * CLAMP 0..1
 * =========================================================
 */


function clampUnit(
    value
) {


    const number =
        normalizeNumber(
            value
        );



    return Math.max(
        0,
        Math.min(
            1,
            number
        )
    );

}





/*
 * =========================================================
 * POSITIVE INTEGER
 * =========================================================
 */


function normalizePositiveInteger(
    value
) {


    const number =
        Math.floor(
            normalizeNumber(
                value
            )
        );



    return number > 0

        ? number

        : 0;

}





/*
 * =========================================================
 * TRACE SUCCESS
 * =========================================================
 */


function resolveTraceSuccess(
    trace
) {


    /*
     * Явный Execution Result
     */


    if(
        typeof trace?.result?.success ===
        "boolean"
    ){

        return trace.result.success;

    }



    /*
     * Явное поле самого Trace
     */


    if(
        typeof trace?.success ===
        "boolean"
    ){

        return trace.success;

    }



    /*
     * Execution statistics
     */


    const completed =

        normalizeNumber(

            trace?.stats?.completed

            ??

            trace?.statistics?.completed

        );



    if(
        completed > 0
    ){

        return true;

    }



    /*
     * Status fallback
     */


    const status =

        String(
            trace?.status || ""
        )
        .trim()
        .toUpperCase();



    if(
        status === "COMPLETED"
    ){

        return true;

    }



    return false;

}





/*
 * =========================================================
 * EXAMPLE SUCCESS
 * =========================================================
 */


function resolveExampleSuccess(
    example
) {


    if(
        typeof example?.success ===
        "boolean"
    ){

        return example.success;

    }



    if(
        typeof example?.result?.success ===
        "boolean"
    ){

        return example.result.success;

    }



    const status =

        String(
            example?.status || ""
        )
        .trim()
        .toUpperCase();



    if(
        status === "COMPLETED"
    ){

        return true;

    }



    if(
        status === "FAILED"
        ||
        status === "NEEDS_CLARIFICATION"
    ){

        return false;

    }



    /*
     * Неизвестный Example больше
     * НЕ считается успешным автоматически.
     */


    return false;

}





/*
 * =========================================================
 * NORMALIZE EXAMPLE
 * =========================================================
 */


function normalizeExample(
    example
) {


    if(
        !example ||
        typeof example !== "object"
    ){

        return null;

    }



    return {


        ...example,


        task:

            String(
                example.task || ""
            )
            .trim(),


        success:

            resolveExampleSuccess(
                example
            )

    };

}





/*
 * =========================================================
 * OCCURRENCES
 * =========================================================
 */


export function getOccurrences(
    trace
) {


    /*
     * Explicit Learning occurrence
     */


    const explicit =

        normalizePositiveInteger(

            trace?.experienceOccurrences

        );



    if(
        explicit > 0
    ){

        return explicit;

    }



    /*
     * Trace statistics
     */


    const traceOccurrences =

        normalizePositiveInteger(

            trace?.stats?.occurrences

            ??

            trace?.statistics?.occurrences

        );



    if(
        traceOccurrences > 0
    ){

        return traceOccurrences;

    }



    /*
     * Aggregated Examples
     */


    if(
        Array.isArray(
            trace?.examples
        )
        &&
        trace.examples.length > 1
    ){

        return trace.examples.length;

    }



    /*
     * Existing Experience usage
     *
     * Runtime statistics относятся
     * к предыдущим использованиям Skill.
     *
     * Текущий Execution является
     * ещё одним occurrence.
     */


    const usedSkills =

        Array.isArray(
            trace?.experienceUsage?.skills
        )

            ? trace.experienceUsage.skills

            : [];



    const usedSkill =

        usedSkills[0] || null;



    if(
        usedSkill
    ){


        const storedOccurrences =

            normalizePositiveInteger(

                usedSkill?.learning?.occurrences

                ??

                usedSkill?.occurrences

            );



        if(
            storedOccurrences > 0
        ){

            return storedOccurrences + 1;

        }



        const successfulRuns =

            normalizePositiveInteger(

                usedSkill
                    ?.statistics
                    ?.successfulRuns

            );



        const failedRuns =

            normalizePositiveInteger(

                usedSkill
                    ?.statistics
                    ?.failedRuns

            );



        const runtimeOccurrences =

            successfulRuns +
            failedRuns;



        if(
            runtimeOccurrences > 0
        ){

            return runtimeOccurrences + 1;

        }

    }



    /*
     * Минимум одно наблюдение:
     * текущий Execution.
     */


    return 1;

}





/*
 * =========================================================
 * EXAMPLES
 * =========================================================
 */


export function getExamples(
    trace
) {


    if(
        Array.isArray(
            trace?.examples
        )
        &&
        trace.examples.length > 0
    ){


        return trace.examples

            .map(
                normalizeExample
            )

            .filter(
                Boolean
            );

    }



    /*
     * Fallback Example строится
     * из самого Execution Trace.
     *
     * success определяется из Trace,
     * а НЕ выставляется true автоматически.
     */


    return [

        {

            task:

                String(
                    trace?.task || ""
                )
                .trim(),


            result:

                trace?.result || null,


            success:

                resolveTraceSuccess(
                    trace
                )

        }

    ];

}





/*
 * =========================================================
 * SUCCESS COUNT
 * =========================================================
 */


export function calculateSuccessCount(
    examples = []
) {


    if(
        !Array.isArray(
            examples
        )
    ){

        return 0;

    }



    return examples.filter(

        item =>
            resolveExampleSuccess(
                item
            ) === true

    )
    .length;

}





/*
 * =========================================================
 * FAILURE COUNT
 * =========================================================
 */


export function calculateFailureCount(
    examples = []
) {


    if(
        !Array.isArray(
            examples
        )
    ){

        return 0;

    }



    return examples.filter(

        item =>
            resolveExampleSuccess(
                item
            ) === false

    )
    .length;

}





/*
 * =========================================================
 * SUCCESS RATE
 * =========================================================
 */


export function calculateSuccessRate(
    examples = []
) {


    if(
        !Array.isArray(
            examples
        )
        ||
        examples.length === 0
    ){

        return 0;

    }



    const successCount =

        calculateSuccessCount(
            examples
        );



    return Number(

        (
            successCount /
            examples.length
        )
        .toFixed(2)

    );

}





/*
 * =========================================================
 * CONFIDENCE
 * =========================================================
 *
 * Формула:
 *
 * Match quality     30%
 * Success rate      40%
 * Repetition        30%
 *
 *
 * Repetition достигает 1.0
 * после пяти подтверждённых наблюдений.
 *
 *
 * Пример:
 *
 * matchScore   = 1
 * successRate  = 1
 * occurrences  = 1
 *
 * confidence = 0.76
 *
 *
 * occurrences = 2
 *
 * confidence = 0.82
 *
 *
 * Это позволяет:
 *
 * - не превращать каждый первый
 *   успешный Execution сразу в Skill;
 *
 * - повышать доверие при повторении.
 *
 * =========================================================
 */


export function calculateExperienceConfidence({

    matchScore = 0,

    successRate = 0,

    occurrences = 1

} = {}) {


    const normalizedMatch =

        clampUnit(
            matchScore
        );



    const normalizedSuccessRate =

        clampUnit(
            successRate
        );



    const normalizedOccurrences =

        Math.max(

            normalizePositiveInteger(
                occurrences
            ),

            1

        );



    const repeatScore =

        Math.min(

            normalizedOccurrences / 5,

            1

        );



    return Number(

        (

            normalizedMatch * 0.30

            +

            normalizedSuccessRate * 0.40

            +

            repeatScore * 0.30

        )
        .toFixed(2)

    );

}





/*
 * =========================================================
 * MATURITY
 * =========================================================
 *
 * Канонический контракт:
 *
 * maturity = число 0..1
 *
 *
 * 1 occurrence  → 0.50
 * 2 occurrences → 0.63
 * 3 occurrences → 0.75
 * 4 occurrences → 0.88
 * 5+            → 1.00
 *
 *
 * Первый качественный опыт уже имеет
 * минимальную зрелость для анализа,
 * но Confidence самостоятельно решает,
 * достаточно ли evidence для AUTO_APPROVE.
 *
 * =========================================================
 */


export function calculateExperienceMaturity(
    occurrences
) {


    const count =

        Math.max(

            normalizePositiveInteger(
                occurrences
            ),

            1

        );



    const maturity =

        0.5

        +

        Math.min(
            count - 1,
            4
        )

        *

        0.125;



    return Number(

        Math.min(
            maturity,
            1
        )
        .toFixed(2)

    );

}





/*
 * =========================================================
 * MATURITY LEVEL
 * =========================================================
 *
 * Человекочитаемый label.
 *
 * Не используется вместо числовой maturity.
 *
 * =========================================================
 */


export function resolveExperienceMaturityLevel(
    occurrences
) {


    const count =

        Math.max(

            normalizePositiveInteger(
                occurrences
            ),

            1

        );



    if(
        count >= 5
    ){

        return "READY";

    }



    if(
        count >= 2
    ){

        return "LEARNING";

    }



    return "NEW";

}
