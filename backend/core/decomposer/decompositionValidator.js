/*
 * =========================================================
 * JESSICA DECOMPOSITION VALIDATOR v1
 * =========================================================
 *
 * AI Decomposition
 *      ↓
 * Structural Validation
 *      ↓
 * Canonical Subtasks
 *
 *
 * Гарантии:
 *
 * - canonical sequential IDs;
 * - максимум MAX_SUBTASKS;
 * - dependsOn только на предыдущие subtasks;
 * - нет self dependency;
 * - нет future dependency;
 * - нет неизвестных dependency IDs;
 * - dependencies deduplicated.
 *
 * =========================================================
 */


export const MAX_SUBTASKS =
    30;


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
 * TEXT
 * =========================================================
 */


function normalizeText(
    value
) {

    return typeof value === "string"

        ? value.trim()

        : "";

}


/*
 * =========================================================
 * MEMORY HINTS
 * =========================================================
 */


function normalizeMemoryHints(
    value
) {

    if(
        !Array.isArray(value)
    ){

        return [];

    }


    return value
        .filter(
            item =>
                item !== null
                &&
                item !== undefined
        )
        .slice(
            0,
            20
        );

}


/*
 * =========================================================
 * DEPENDENCY ID
 * =========================================================
 */


function normalizeDependencyId(
    value
) {

    const number =
        Number(value);


    if(
        !Number.isInteger(number)
        ||
        number < 1
    ){

        return null;

    }


    return number;

}


/*
 * =========================================================
 * DEPENDENCIES
 * =========================================================
 */


function normalizeDependencies(
    value,
    currentId
) {

    const source =

        Array.isArray(value)

            ? value

            : (
                value === null
                ||
                value === undefined

                    ? []

                    : [value]
            );


    const result = [];


    for(
        const rawId
        of source
    ){

        const dependencyId =

            normalizeDependencyId(
                rawId
            );


        if(
            dependencyId === null
        ){

            return {

                success:
                    false,

                reason:

                    `Некорректный dependency id у подзадачи ${currentId}`

            };

        }


        /*
         * Только previous dependencies.
         *
         * Это одновременно запрещает:
         *
         * - self dependency;
         * - future dependency;
         * - cycles.
         */


        if(
            dependencyId >= currentId
        ){

            return {

                success:
                    false,

                reason:

                    `Подзадача ${currentId} зависит от недопустимой подзадачи ${dependencyId}`

            };

        }


        if(
            !result.includes(
                dependencyId
            )
        ){

            result.push(
                dependencyId
            );

        }

    }


    return {

        success:
            true,

        dependencies:
            result

    };

}


/*
 * =========================================================
 * VALIDATE
 * =========================================================
 */


export function validateDecomposition({

    data,

    originalTask,

    memoryContext

} = {}) {

    if(
        !isObject(
            data
        )
    ){

        return {

            success:
                false,

            text:
                "Некорректная структура decomposition"

        };

    }


    if(
        !Array.isArray(
            data.subtasks
        )
    ){

        return {

            success:
                false,

            text:
                "Нет списка подзадач"

        };

    }


    if(
        data.subtasks.length === 0
    ){

        return {

            success:
                false,

            text:
                "Пустой список подзадач"

        };

    }


    if(
        data.subtasks.length > MAX_SUBTASKS
    ){

        return {

            success:
                false,

            text:
                `Количество подзадач превышает лимит ${MAX_SUBTASKS}`

        };

    }


    const subtasks = [];


    for(
        let index = 0;
        index < data.subtasks.length;
        index++
    ){

        const item =

            data.subtasks[index];


        if(
            !isObject(
                item
            )
        ){

            return {

                success:
                    false,

                text:

                    `Некорректная подзадача ${index + 1}`

            };

        }


        const canonicalId =
            index + 1;


        /*
         * Если AI прислал id,
         * он обязан совпадать
         * с canonical ordering.
         */


        if(
            item.id !== undefined
            &&
            item.id !== null
        ){

            const suppliedId =

                Number(
                    item.id
                );


            if(
                !Number.isInteger(
                    suppliedId
                )
                ||
                suppliedId !== canonicalId
            ){

                return {

                    success:
                        false,

                    text:

                        `Некорректный id подзадачи ${canonicalId}`

                };

            }

        }


        const text =

            normalizeText(
                item.text
            );


        if(
            !text
        ){

            return {

                success:
                    false,

                text:

                    `Пустая подзадача ${canonicalId}`

            };

        }


        const dependencyResult =

            normalizeDependencies(

                item.dependsOn

                ??

                item.dependencies

                ??

                [],

                canonicalId

            );


        if(
            !dependencyResult.success
        ){

            return {

                success:
                    false,

                text:
                    dependencyResult.reason

            };

        }


        subtasks.push({

            id:
                canonicalId,

            text,

            dependsOn:
                dependencyResult.dependencies,

            contributesToFinalAnswer:

                item.contributesToFinalAnswer
                !==
                false,

            memoryHints:

                normalizeMemoryHints(
                    item.memoryHints
                )

        });

    }


    return {

        success:
            true,

        decomposition: {

            originalTask:

                normalizeText(
                    originalTask
                ),

            isComplex:

                subtasks.length > 1,

            memoryContext:

                memoryContext

                ??

                null,

            subtasks

        }

    };

                  }
