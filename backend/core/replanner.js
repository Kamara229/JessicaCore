import {
    createPlan
} from "./planner.js";

import {
    buildRetryContext
} from "./replanner/retryContext.js";


export async function replanTask(
    task,
    previousPlan,
    failureResult,
    previousRunResult,
    planningContext = {}
) {

    const retryContext =
        buildRetryContext({
            planningContext,
            previousPlan,
            previousRunResult,
            failureResult
        });


    console.log(
        "Jessica Replanner context:",
        JSON.stringify(
            retryContext.metadata?.retry || {}
        )
    );


    try {

        const result =
            await createPlan(
                task,
                retryContext
            );


        if (
            !result?.success ||
            !result?.plan
        ) {

            return {
                success: false,
                reason:
                    result?.text ||
                    "Planner не смог создать альтернативный маршрут"
            };

        }


        console.log(
            "Jessica Replanner new plan:",
            JSON.stringify(
                result.plan
            )
        );


        return {
            success: true,
            plan: result.plan,
            context:
                result?.context ||
                retryContext
        };

    } catch (error) {

        console.error(
            "Jessica Replanner error:",
            error
        );


        return {
            success: false,
            reason:
                error?.message ||
                "Ошибка Replanner"
        };

    }

}
