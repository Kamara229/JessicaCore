/*
 * =========================================================
 * JESSICA REFERENCE VALIDATOR
 * =========================================================
 *
 * Проверяет зависимости между шагами Planner.
 *
 * Поддерживает ссылки вида:
 *
 * {
 *   "$from": "search",
 *   "path": "data.results.0.url"
 * }
 *
 * Ссылка может вести только на уже предыдущий шаг.
 */


/*
 * =========================================================
 * VALIDATE REFERENCE OBJECT
 * =========================================================
 */


function validateReferenceObject(
    value,
    previousStepIds
) {

    const from =
        typeof value?.$from === "string"
            ? value.$from.trim()
            : "";


    if (!from) {

        return {
            success: false,

            text:
                "Некорректный $from"
        };

    }


    if (
        !previousStepIds.has(
            from
        )
    ) {

        return {
            success: false,

            text:
                `Ссылка ведёт на неизвестный или будущий шаг: ${from}`
        };

    }


    if (
        value.path !== undefined &&
        (
            typeof value.path !== "string" ||
            !value.path.trim()
        )
    ) {

        return {
            success: false,

            text:
                `Некорректный path для шага ${from}`
        };

    }


    return {
        success: true
    };

}


/*
 * =========================================================
 * VALIDATE REFERENCES
 * =========================================================
 */


export function validateReferences(
    value,
    previousStepIds
) {

    if (
        value === null ||
        value === undefined ||
        typeof value !== "object"
    ) {

        return {
            success: true
        };

    }


    /*
     * Ссылка на предыдущий шаг.
     */
    if (
        !Array.isArray(
            value
        ) &&
        Object.prototype.hasOwnProperty.call(
            value,
            "$from"
        )
    ) {

        return validateReferenceObject(
            value,
            previousStepIds
        );

    }


    /*
     * Массив.
     */
    if (
        Array.isArray(
            value
        )
    ) {

        for (
            const item
            of value
        ) {

            const validation =
                validateReferences(
                    item,
                    previousStepIds
                );


            if (
                !validation.success
            ) {

                return validation;

            }

        }


        return {
            success: true
        };

    }


    /*
     * Обычный объект.
     */
    for (
        const item
        of Object.values(
            value
        )
    ) {

        const validation =
            validateReferences(
                item,
                previousStepIds
            );


        if (
            !validation.success
        ) {

            return validation;

        }

    }


    return {
        success: true
    };

}
