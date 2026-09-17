/*
 * =========================================================
 * JESSICA TASK RUNNER
 * PLAN RUNTIME VALIDATOR
 * =========================================================
 *
 * Проверяет план непосредственно перед выполнением.
 *
 *
 * НЕ:
 *
 * - выполняет инструменты;
 * - вызывает Planner;
 * - делает Replan;
 * - проверяет semantic correctness.
 *
 *
 * Основной Plan Validator работает раньше.
 *
 * Этот модуль является последней runtime-защитой.
 *
 * =========================================================
 */


const MAX_STEPS =
    15;


/*
 * =========================================================
 * STEP ID
 * =========================================================
 */


export function getStepId(
    step,
    index
) {

    if (
        typeof step?.id === "string" &&
        step.id.trim()
    ) {

        return step.id.trim();

    }


    return `step_${index + 1}`;
}



/*
 * =========================================================
 * FAILURE
 * =========================================================
 */


function buildFailure(
    text,
    failureType
) {

    return {

        success:
            false,

        shouldRetry:
            false,

        stage:
            "runner",

        failureType,

        text

    };

}



/*
 * =========================================================
 * UNIQUE IDS
 * =========================================================
 */


function validateStepIds(
    steps
) {

    const ids =
        new Set();


    for (
        let index = 0;
        index < steps.length;
        index++
    ) {

        const id =
            getStepId(
                steps[index],
                index
            );


        if (
            ids.has(id)
        ) {

            return buildFailure(

                `В плане повторяется id шага: ${id}`,

                "duplicate-step-id"

            );

        }


        ids.add(id);

    }


    return {
        success:
            true
    };

}



/*
 * =========================================================
 * VALIDATE PLAN
 * =========================================================
 */


export function validatePlanForExecution(
    plan
) {

    if (
        !plan ||
        typeof plan !== "object" ||
        Array.isArray(plan)
    ) {

        return buildFailure(

            "TaskRunner получил некорректный план",

            "invalid-plan"

        );

    }



    if (
        !Array.isArray(
            plan.steps
        )
    ) {

        return buildFailure(

            "В плане отсутствует массив steps",

            "missing-steps"

        );

    }



    /*
     * План не требует инструментов.
     */


    if (
        plan.requiresTools === false
    ) {

        return {

            success:
                true,

            noTools:
                true

        };

    }



    if (
        plan.steps.length === 0
    ) {

        return buildFailure(

            "План требует инструменты, но не содержит шагов",

            "empty-tool-plan"

        );

    }



    if (
        plan.steps.length >
        MAX_STEPS
    ) {

        return buildFailure(

            (
                `План содержит слишком много шагов: ` +
                `${plan.steps.length}. Максимум: ${MAX_STEPS}.`
            ),

            "too-many-steps"

        );

    }



    const idsValidation =
        validateStepIds(
            plan.steps
        );


    if (
        !idsValidation.success
    ) {

        return idsValidation;

    }



    return {

        success:
            true,

        noTools:
            false

    };

}
