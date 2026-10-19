import { buildKnownLicenseComparisonPlan } from "../planner/knownLicenseComparison.js";

/*
 * =========================================================
 * JESSICA NATIVE DECOMPOSER v1
 * =========================================================
 *
 * Детерминированная декомпозиция только
 * для явно последовательных пользовательских задач.
 *
 *
 * Примеры:
 *
 * "Найди X. Затем рассчитай Y."
 * "Find X. Then calculate Y."
 *
 *
 * Не пытается понимать произвольную сложную задачу.
 * Если структура неочевидна — возвращает handled=false
 * и управление остаётся AI Decomposer.
 *
 * =========================================================
 */


const MAX_NATIVE_SUBTASKS =
    4;


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .replace(
            /\s+/g,
            " "
        )
        .trim();

}


function normalizeMemoryHints(
    memoryContext
) {

    const skills =

        Array.isArray(
            memoryContext?.skills
        )

            ? memoryContext.skills

            : [];


    return skills

        .map(
            skill => ({

                id:
                    skill?.id
                    ||
                    skill?.skillId
                    ||
                    null,

                name:
                    normalizeText(
                        skill?.name
                    ),

                workflow:

                    Array.isArray(
                        skill?.workflow
                    )

                        ? [
                            ...skill.workflow
                        ]

                        : [],

                constraints:

                    Array.isArray(
                        skill?.constraints
                    )

                        ? [
                            ...skill.constraints
                        ]

                        : []

            })
        )

        .filter(
            skill =>
                skill.id
                ||
                skill.name
        );

}


/*
 * =========================================================
 * EXPLICIT SEQUENCE
 * =========================================================
 */


function splitExplicitSequence(
    task
) {

    const text =
        normalizeText(
            task
        );


    if(!text){

        return [];

    }


    /*
     * Нужен именно явный переход к следующей цели.
     *
     * Разделитель сохраняет смысл второй части,
     * но не переносится как отдельная инструкция.
     */


    const parts =

        text
            .split(
                /(?:[.!?]\s*)?(?=(?:затем|после\s+этого|далее|после\s+чего)(?:\s|[,;:]|$)|\b(?:then|after\s+that|next)\b)/giu
            )
            .map(
                item =>
                    item
                        .replace(
                            /^(?:затем|после\s+этого|далее|после\s+чего|then|after\s+that|next)[,:]?\s*/iu,
                            ""
                        )
                        .replace(
                            /^[.!?\s]+|[\s]+$/g,
                            ""
                        )
            )
            .filter(
                item =>
                    item.length >= 12
            );


    if(
        parts.length < 2
        ||
        parts.length > MAX_NATIVE_SUBTASKS
    ){

        return [];

    }


    return parts;

}


/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export function tryNativeDecomposition({

    task,

    memoryContext = null

} = {}) {

    // Keep a two-source comparison as one semantic task.
    // Splitting at "then" loses the URLs needed for the comparison.
    const input = String(task || "");
    const urls = input.match(/https?:\/\/[^\s]+/gi) || [];
    if (urls.length === 2 && /сравн|сопостав|compare/i.test(input)) {
        return {
            handled: true,
            source: "native-source-pair",
            decomposition: {
                originalTask: input,
                isComplex: false,
                memoryContext,
                subtasks: [{
                    id: 1,
                    text: input,
                    dependsOn: [],
                    contributesToFinalAnswer: true,
                    memoryHints: normalizeMemoryHints(memoryContext)
                }]
            }
        };
    }

    // Source retrieval and comparison are ONE task when the user names
    // two known licenses. Splitting the names away from the comparison
    // loses identifiers and forces needless AI replanning.
    if (buildKnownLicenseComparisonPlan(input)) {
        return {
            handled: true,
            source: "native-known-license-cohesive",
            decomposition: {
                originalTask: input,
                isComplex: false,
                memoryContext,
                subtasks: [{
                    id: 1,
                    text: input,
                    dependsOn: [],
                    contributesToFinalAnswer: true,
                    memoryHints: normalizeMemoryHints(memoryContext)
                }]
            }
        };
    }

    const parts =

        splitExplicitSequence(
            task
        );


    if(
        parts.length < 2
    ){

        return {

            handled:
                false,

            decomposition:
                null,

            source:
                null

        };

    }


    const memoryHints =

        normalizeMemoryHints(
            memoryContext
        );


    const subtasks =

        parts.map(
            (
                text,
                index
            ) => ({

                id:
                    index + 1,

                text,

                dependsOn:

                    index === 0

                        ? []

                        : [
                            index
                        ],

                contributesToFinalAnswer:
                    true,

                memoryHints: [
                    ...memoryHints
                ]

            })
        );


    return {

        handled:
            true,

        source:
            "native-sequence",

        decomposition: {

            originalTask:
                normalizeText(
                    task
                ),

            isComplex:
                true,

            memoryContext,

            subtasks

        }

    };

}
