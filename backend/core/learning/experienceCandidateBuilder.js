/*
 * =========================================================
 * JESSICA EXPERIENCE CANDIDATE BUILDER v2
 * =========================================================
 *
 * Создаёт Learning Candidate
 * из Execution Trace и Learning Evidence.
 *
 *
 * Flow:
 *
 * Pattern / Existing Skill
 *          +
 * Execution Trace
 *          +
 * Confidence Metrics
 *          ↓
 * Candidate Builder
 *          ↓
 * Full Experience Candidate
 *
 *
 * Возможные Candidate:
 *
 * NEW_SKILL
 *
 * SKILL_IMPROVEMENT
 *
 *
 * Ответственность:
 *
 * - сформировать полный Candidate;
 * - сохранить Evidence;
 * - сохранить Learning Metrics;
 * - подготовить полноценный
 *   proposed Experience;
 * - при Improvement сохранить
 *   знания существующего Skill.
 *
 *
 * НЕ:
 *
 * - сохраняет Skill;
 * - вызывает AI;
 * - ищет похожие Candidate;
 * - принимает Learning Decision;
 * - выполняет AUTO_APPROVE;
 * - создаёт новую версию в Supabase.
 *
 * =========================================================
 */


import {
    calculateSuccessCount,
    calculateFailureCount,
    calculateSuccessRate,
    resolveExperienceMaturityLevel
} from "./experienceConfidence.js";





/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function isObject(
    value
) {


    return (

        value &&

        typeof value === "object" &&

        !Array.isArray(value)

    );

}





function normalizeArray(
    value
) {


    return Array.isArray(value)

        ? [...value]

        : [];

}





function normalizeText(
    value
) {


    return String(
        value || ""
    )
    .trim();

}





function normalizeNumber(
    value
) {


    const number =
        Number(value);



    return Number.isFinite(number)

        ? number

        : 0;

}





function normalizeUnit(
    value
) {


    return Math.max(

        0,

        Math.min(

            1,

            normalizeNumber(
                value
            )

        )

    );

}





/*
 * =========================================================
 * NORMALIZE STRING ARRAY
 * =========================================================
 */


function normalizeStringArray(
    value
) {


    if(
        !Array.isArray(
            value
        )
    ){

        return [];

    }



    const result = [];



    for(
        const item
        of value
    ){


        const normalized =

            normalizeText(
                item
            );



        if(
            !normalized
        ){

            continue;

        }



        if(
            !result.includes(
                normalized
            )
        ){

            result.push(
                normalized
            );

        }

    }



    return result;

}





/*
 * =========================================================
 * MERGE STRING ARRAYS
 * =========================================================
 */


function mergeStringArrays(
    ...arrays
) {


    return normalizeStringArray(

        arrays.flatMap(

            value =>
                Array.isArray(value)
                    ? value
                    : []

        )

    );

}





/*
 * =========================================================
 * TRACE SUCCESS
 * =========================================================
 */


function resolveTraceSuccess(
    trace
) {


    if(
        typeof trace?.result?.success ===
        "boolean"
    ){

        return trace.result.success;

    }



    if(
        typeof trace?.success ===
        "boolean"
    ){

        return trace.success;

    }



    const completed =

        Number(

            trace?.stats?.completed

            ??

            trace?.statistics?.completed

            ??

            0

        );



    return (

        Number.isFinite(completed)

        &&

        completed > 0

    );

}





/*
 * =========================================================
 * BUILD TRACE EXAMPLE
 * =========================================================
 */


function buildTraceExample(
    trace
) {


    return {


        task:

            normalizeText(
                trace?.task
            ),



        result:

            trace?.result || null,



        success:

            resolveTraceSuccess(
                trace
            ),



        traceId:

            trace?.id || null,



        createdAt:

            new Date()
                .toISOString()


    };

}





/*
 * =========================================================
 * BUILD EXAMPLES
 * =========================================================
 */


function buildExamples(
    trace
) {


    if(
        Array.isArray(
            trace?.examples
        )
        &&
        trace.examples.length > 0
    ){

        return [

            ...trace.examples

        ];

    }



    return [

        buildTraceExample(
            trace
        )

    ];

}





/*
 * =========================================================
 * EXAMPLE KEY
 * =========================================================
 */


function buildExampleKey(
    example
) {


    if(
        !example ||
        typeof example !== "object"
    ){

        return "";

    }



    if(
        example.traceId
    ){

        return (

            "trace:"

            +

            String(
                example.traceId
            )

        );

    }



    const task =

        normalizeText(
            example.task
        );



    let result = "";



    try {


        result =

            JSON.stringify(
                example.result ?? null
            );


    }catch(error){


        result =

            String(
                example.result || ""
            );

    }



    return (

        task

        +

        "::"

        +

        result

    );

}





/*
 * =========================================================
 * MERGE EXAMPLES
 * =========================================================
 */


function mergeExamples(
    ...exampleSets
) {


    const result = [];

    const known = new Set();



    for(
        const set
        of exampleSets
    ){


        if(
            !Array.isArray(
                set
            )
        ){

            continue;

        }



        for(
            const example
            of set
        ){


            if(
                !example ||
                typeof example !== "object"
            ){

                continue;

            }



            const key =

                buildExampleKey(
                    example
                );



            if(
                key &&
                known.has(
                    key
                )
            ){

                continue;

            }



            if(
                key
            ){

                known.add(
                    key
                );

            }



            result.push({

                ...example

            });

        }

    }



    return result;

}





/*
 * =========================================================
 * RESOLVE SKILL DATA
 * =========================================================
 *
 * Поддерживаем:
 *
 * Skill
 *
 * и
 *
 * {
 *     experience: Skill
 * }
 *
 * чтобы Builder не зависел
 * от transport wrapper.
 *
 * =========================================================
 */


function resolveSkillData(
    skill
) {


    if(
        !isObject(
            skill
        )
    ){

        return null;

    }



    if(
        isObject(
            skill.experience
        )
    ){

        return skill.experience;

    }



    return skill;

}





/*
 * =========================================================
 * BUILD LEARNING METRICS
 * =========================================================
 */


function buildLearningMetrics({

    examples,

    occurrences,

    maturity,

    confidence

}) {


    const normalizedExamples =

        Array.isArray(
            examples
        )

            ? examples

            : [];



    const successCount =

        calculateSuccessCount(
            normalizedExamples
        );



    const failureCount =

        calculateFailureCount(
            normalizedExamples
        );



    const successRate =

        calculateSuccessRate(
            normalizedExamples
        );



    const normalizedOccurrences =

        Math.max(

            Math.floor(
                normalizeNumber(
                    occurrences
                )
            ),

            normalizedExamples.length,

            1

        );



    return {


        occurrences:

            normalizedOccurrences,



        successCount,



        failureCount,



        successRate,



        maturity:

            normalizeUnit(
                maturity
            ),



        maturityLevel:

            resolveExperienceMaturityLevel(
                normalizedOccurrences
            ),



        confidence:

            normalizeUnit(
                confidence
            )

    };

}





/*
 * =========================================================
 * BUILD NEW SKILL CANDIDATE
 * =========================================================
 */


export function buildNewSkillCandidate({

    pattern,

    trace,

    confidence,

    maturity,

    occurrences

}) {


    if(
        !isObject(
            pattern
        )
    ){

        return null;

    }



    const skillId =

        normalizeText(

            pattern.id

            ||

            pattern.skillId

        );



    const name =

        normalizeText(
            pattern.name
        );



    if(
        !skillId ||
        !name
    ){

        return null;

    }



    const examples =

        buildExamples(
            trace
        );



    const metrics =

        buildLearningMetrics({

            examples,

            occurrences,

            maturity,

            confidence

        });



    return {


        /*
         * =================================================
         * IDENTITY
         * =================================================
         */


        skillId,



        name,



        category:

            normalizeText(
                pattern.category
            )

            ||

            "general",



        description:

            normalizeText(
                pattern.description
            ),



        /*
         * =================================================
         * KNOWLEDGE
         * =================================================
         */


        workflow:

            normalizeArray(
                pattern.workflow
            ),



        validationRules:

            normalizeStringArray(
                pattern.validationRules
            ),



        triggerPatterns:

            normalizeStringArray(

                pattern.triggerPatterns

                ||

                pattern.keywords

            ),



        keywords:

            normalizeStringArray(
                pattern.keywords
            ),



        tags:

            normalizeStringArray(
                pattern.tags
            ),



        constraints:

            normalizeStringArray(
                pattern.constraints
            ),



        strategy:

            normalizeStringArray(
                pattern.strategy
            ),



        sourcePriority:

            normalizeStringArray(
                pattern.sourcePriority
            ),



        successfulPatterns:

            normalizeStringArray(
                pattern.successfulPatterns
            ),



        failurePatterns:

            normalizeStringArray(
                pattern.failurePatterns
            ),



        avoidPatterns:

            normalizeStringArray(
                pattern.avoidPatterns
            ),



        requiredTools:

            normalizeStringArray(
                pattern.requiredTools
            ),



        /*
         * =================================================
         * EVIDENCE
         * =================================================
         */


        examples,



        /*
         * =================================================
         * LEARNING METRICS
         * =================================================
         */


        ...metrics,



        /*
         * =================================================
         * META
         * =================================================
         */


        candidateType:

            "NEW_SKILL",



        source:

            "execution-trace"


    };

}





/*
 * =========================================================
 * BUILD SKILL IMPROVEMENT CANDIDATE
 * =========================================================
 *
 * Improvement Candidate должен содержать
 * ПОЛНОЦЕННОЕ состояние Experience,
 * а не только новый Example.
 *
 *
 * Важно:
 *
 * skillId намеренно НЕ записывается
 * в корень Candidate.
 *
 * Текущий Learning Proposal Resolver
 * использует наличие candidate.skillId
 * как признак NEW_SKILL.
 *
 * Existing Skill передаётся отдельно:
 *
 * candidate.skills
 *        ↓
 * Learning Router
 *        ↓
 * event.payload.skills
 *
 * =========================================================
 */


export function buildSkillImprovementCandidate({

    skills,

    trace,

    confidence,

    maturity,

    occurrences

}) {


    const skillList =

        Array.isArray(
            skills
        )

            ? skills.filter(
                isObject
            )

            : [];



    if(
        skillList.length === 0
    ){

        return null;

    }



    const rawExistingSkill =

        skillList[0];



    const existingSkill =

        resolveSkillData(
            rawExistingSkill
        );



    if(
        !existingSkill
    ){

        return null;

    }



    const newExamples =

        buildExamples(
            trace
        );



    const examples =

        mergeExamples(

            normalizeArray(
                existingSkill.examples
            ),

            newExamples

        );



    const metrics =

        buildLearningMetrics({

            examples,

            occurrences,

            maturity,

            confidence

        });



    const targetSkillId =

        normalizeText(

            existingSkill.id

            ||

            existingSkill.skillId

            ||

            rawExistingSkill.id

            ||

            rawExistingSkill.skillId

        )

        ||

        null;



    /*
     * Сейчас Improvement является
     * evidence reinforcement:
     *
     * существующее Knowledge сохраняется,
     * а новый Execution добавляет Evidence.
     *
     *
     * Позже AI Improvement Analyzer
     * сможет дополнительно менять:
     *
     * workflow
     * constraints
     * validationRules
     * failurePatterns
     * strategy
     *
     * Но даже до появления AI merge
     * Candidate уже является полноценным
     * Experience и проходит Reviewer.
     */


    return {


        /*
         * Existing Skill transport
         */


        skills:

            skillList,



        targetSkillId,



        baseVersion:

            Number(
                existingSkill.version || 0
            )

            ||

            null,



        /*
         * =================================================
         * IDENTITY
         * =================================================
         */


        name:

            normalizeText(
                existingSkill.name
            )

            ||

            (
                targetSkillId

                    ?

                    `Jessica Skill ${targetSkillId}`

                    :

                    "Jessica Experience Skill"
            ),



        category:

            normalizeText(
                existingSkill.category
            )

            ||

            "general",



        description:

            normalizeText(
                existingSkill.description
            ),



        /*
         * =================================================
         * KNOWLEDGE
         * =================================================
         */


        workflow:

            normalizeArray(
                existingSkill.workflow
            ),



        validationRules:

            normalizeStringArray(
                existingSkill.validationRules
            ),



        triggerPatterns:

            mergeStringArrays(

                existingSkill.triggerPatterns,

                existingSkill.keywords

            ),



        keywords:

            normalizeStringArray(
                existingSkill.keywords
            ),



        tags:

            normalizeStringArray(
                existingSkill.tags
            ),



        constraints:

            normalizeStringArray(
                existingSkill.constraints
            ),



        strategy:

            normalizeStringArray(
                existingSkill.strategy
            ),



        sourcePriority:

            normalizeStringArray(
                existingSkill.sourcePriority
            ),



        successfulPatterns:

            normalizeStringArray(
                existingSkill.successfulPatterns
            ),



        failurePatterns:

            normalizeStringArray(
                existingSkill.failurePatterns
            ),



        avoidPatterns:

            normalizeStringArray(
                existingSkill.avoidPatterns
            ),



        requiredTools:

            normalizeStringArray(
                existingSkill.requiredTools
            ),



        /*
         * =================================================
         * EVIDENCE
         * =================================================
         */


        examples,



        /*
         * =================================================
         * LEARNING METRICS
         * =================================================
         */


        ...metrics,



        /*
         * =================================================
         * IMPROVEMENT META
         * =================================================
         */


        candidateType:

            "SKILL_IMPROVEMENT",



        improvementType:

            "EVIDENCE_REINFORCEMENT",



        source:

            "execution-trace"


    };

}
