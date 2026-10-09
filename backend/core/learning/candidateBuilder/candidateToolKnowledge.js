/*
 * =========================================================
 * JESSICA CANDIDATE TOOL KNOWLEDGE v2
 * =========================================================
 *
 * Определяет обязательные Tools Experience
 * на основании независимых успешных Execution.
 *
 *
 * Главный принцип:
 *
 * executedTools
 * = инструменты конкретного выполнения.
 *
 * requiredTools
 * = инструменты, устойчиво необходимые
 *   в нескольких подтверждённых выполнениях.
 *
 *
 * requiredTools НЕ строится:
 *
 * - из Planner steps;
 * - из текста validationRules;
 * - из одного случайного Execution;
 * - простым union всех использованных Tools.
 *
 *
 * Алгоритм:
 *
 * verified successful examples
 *        ↓
 * actual used tools
 *        ↓
 * минимум 2 независимых execution
 *        ↓
 * intersection
 *        ↓
 * requiredTools
 *
 * =========================================================
 */


import {
    normalizeStringArray,
    mergeStringArrays
} from "./candidateUtils.js";


/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const MIN_REQUIRED_TOOL_EVIDENCE =
    2;


/*
 * =========================================================
 * OBJECT
 * =========================================================
 */


function isObject(
    value
) {

    return Boolean(

        value
        &&
        typeof value === "object"
        &&
        !Array.isArray(value)

    );

}


/*
 * =========================================================
 * EXAMPLE IDENTITY
 * =========================================================
 *
 * Нужна для защиты от повторной обработки
 * одного и того же Execution.
 *
 * =========================================================
 */


function buildEvidenceIdentity(
    example
) {

    if(
        !isObject(example)
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


    const result =

        isObject(
            example.result
        )

            ? example.result

            : {};


    if(
        result?.executionMeta?.traceId
    ){

        return (
            "execution-trace:"
            +
            String(
                result.executionMeta.traceId
            )
        );

    }


    if(
        result?.executionMeta?.executionId
    ){

        return (
            "execution:"
            +
            String(
                result.executionMeta.executionId
            )
        );

    }


    const task =

        String(
            example.task || ""
        )
        .trim();


    const createdAt =

        String(
            example.createdAt || ""
        )
        .trim();


    if(
        task
        &&
        createdAt
    ){

        return (
            "fallback:"
            +
            task
            +
            "::"
            +
            createdAt
        );

    }


    return "";

}


/*
 * =========================================================
 * VERIFIED SUCCESS
 * =========================================================
 */


function isVerifiedSuccessfulExample(
    example
) {

    if(
        !isObject(example)
    ){

        return false;

    }


    if(
        example.success !== true
    ){

        return false;

    }


    const result =

        isObject(
            example.result
        )

            ? example.result

            : null;


    if(
        !result
    ){

        return false;

    }


    /*
     * Execution Layer должен подтвердить Result.
     */


    if(
        result.verified !== true
    ){

        return false;

    }


    const validation =

        isObject(
            result.validation
        )

            ? result.validation

            : null;


    if(
        !validation
        ||
        validation.valid !== true
    ){

        return false;

    }


    /*
     * Корректный NO_VERIFIED_RESULT
     * не является доказательством
     * обязательного Tool contract.
     */


    if(
        validation.outcomeType
        &&
        validation.outcomeType !== "result"
    ){

        return false;

    }


    return true;

}


/*
 * =========================================================
 * EXAMPLE TOOLS
 * =========================================================
 *
 * Новый формат:
 *
 * example.executedTools
 *
 *
 * Legacy fallback:
 *
 * example.result.executionMeta.usedTools
 *
 *
 * Это позволяет использовать старую
 * накопленную историю Experience.
 *
 * =========================================================
 */


export function resolveExampleExecutedTools(
    example
) {

    if(
        !isObject(example)
    ){

        return [];

    }


    return mergeStringArrays(

        example.executedTools,

        example
            ?.result
            ?.executionMeta
            ?.usedTools

    );

}


/*
 * =========================================================
 * TOOL EVIDENCE
 * =========================================================
 */


function collectToolEvidence(
    examples
) {

    const source =

        Array.isArray(examples)

            ? examples

            : [];


    const knownExecutions =
        new Set();


    const evidence = [];


    for(
        const example
        of source
    ){

        if(
            !isVerifiedSuccessfulExample(
                example
            )
        ){

            continue;

        }


        const identity =

            buildEvidenceIdentity(
                example
            );


        /*
         * Для machine contract
         * нам нужна независимая
         * Execution identity.
         */


        if(
            !identity
        ){

            continue;

        }


        if(
            knownExecutions.has(
                identity
            )
        ){

            continue;

        }


        const tools =

            resolveExampleExecutedTools(
                example
            );


        /*
         * Пустой список здесь считаем
         * отсутствием надёжной telemetry,
         * а не доказательством того,
         * что Tools не нужны.
         */


        if(
            tools.length === 0
        ){

            continue;

        }


        knownExecutions.add(
            identity
        );


        evidence.push({

            identity,

            tools

        });

    }


    return evidence;

}


/*
 * =========================================================
 * INTERSECTION
 * =========================================================
 */


function intersectToolSets(
    evidence
) {

    if(
        !Array.isArray(evidence)
        ||
        evidence.length === 0
    ){

        return [];

    }


    let required =

        normalizeStringArray(
            evidence[0]?.tools
        );


    for(
        let index = 1;
        index < evidence.length;
        index++
    ){

        const current =

            new Set(

                normalizeStringArray(
                    evidence[index]?.tools
                )

            );


        required =

            required.filter(
                tool =>
                    current.has(tool)
            );


        if(
            required.length === 0
        ){

            break;

        }

    }


    return required;

}


/*
 * =========================================================
 * INFER REQUIRED TOOLS
 * =========================================================
 */


export function inferRequiredToolsFromExamples(

    examples,

    {
        minEvidence =
            MIN_REQUIRED_TOOL_EVIDENCE
    } = {}

) {

    const evidence =

        collectToolEvidence(
            examples
        );


    const requiredEvidence =

        Math.max(

            Number(
                minEvidence
            )
            || MIN_REQUIRED_TOOL_EVIDENCE,

            2

        );


    /*
     * Один успешный запуск ещё
     * не создаёт обязательный contract.
     */


    if(
        evidence.length
        <
        requiredEvidence
    ){

        return {

            requiredTools:
                [],

            evidenceCount:
                evidence.length,

            sufficientEvidence:
                false

        };

    }


    const requiredTools =

        intersectToolSets(
            evidence
        );


    return {

        requiredTools,

        evidenceCount:
            evidence.length,

        sufficientEvidence:
            true

    };

}


/*
 * =========================================================
 * NEW SKILL
 * =========================================================
 */


export function resolveNewSkillRequiredTools({

    examples

} = {}) {

    return inferRequiredToolsFromExamples(
        examples
    );

}


/*
 * =========================================================
 * IMPROVEMENT
 * =========================================================
 *
 * Уже сформированный non-empty contract
 * автоматически не расширяем и не меняем.
 *
 *
 * Legacy Skill с requiredTools=[]
 * может получить bootstrap только тогда,
 * когда накоплено минимум два
 * независимых verified Execution.
 *
 * =========================================================
 */


export function resolveImprovementRequiredTools({

    existingSkill,

    examples

} = {}) {

    const existingTools =

        normalizeStringArray(
            existingSkill?.requiredTools
        );


    /*
     * Existing machine contract стабилен.
     *
     * Его изменение позже должен выполнять
     * отдельный semantic Improvement Policy.
     */


    if(
        existingTools.length > 0
    ){

        return {

            requiredTools:
                existingTools,

            bootstrapped:
                false,

            evidenceCount:
                0,

            sufficientEvidence:
                true

        };

    }


    const inferred =

        inferRequiredToolsFromExamples(
            examples
        );


    return {

        requiredTools:

            inferred.requiredTools,


        bootstrapped:

            inferred.sufficientEvidence === true
            &&
            inferred.requiredTools.length > 0,


        evidenceCount:

            inferred.evidenceCount,


        sufficientEvidence:

            inferred.sufficientEvidence

    };

            }
