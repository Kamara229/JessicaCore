/*
 * =========================================================
 * JESSICA EXECUTION CYCLE v8
 * =========================================================
 */


import {
    createExecutionContext
} from "./executionContext.js";


import {
    executeExecutionStep
} from "./executionStepRunner.js";


import {
    handleExecutionFailure,
    FAILURE_ACTION
} from "./executionFailureHandler.js";


import {
    createAlternativePlan,
    applyAlternativePlan
} from "./replanCoordinator.js";


import {
    buildTerminalResult
} from "./executionTerminal.js";


import {
    createExecutionTrace,
    updateTraceFromResult,
    finishExecutionTrace,
    addTraceEvent,
    addTraceAttempt,
    addTraceReplan
} from "../trace/executionTrace.js";


import {
    MAX_EXECUTION_ATTEMPTS
} from "./retryPolicy.js";









function extractExperience(
    plan,
    planningContext
) {

    const experience =
        planningContext?.experience;



    return {

        found:
            experience?.found === true,


        source:
            experience?.source || null,


        confidence:
            Number(
                experience?.confidence || 0
            ),


        skills:

            Array.isArray(
                plan?.experience?.skills
            )

                ? plan.experience.skills

                : [],


        context:
            experience || null

    };

}









function createJessicaContext({

    task,

    plan,

    planningContext

}) {


    const context =
        createExecutionContext({

            task,

            plan,

            planningContext,


            experience:

                extractExperience(
                    plan,
                    planningContext
                )

        });



    context.initialPlan =
        plan;



    return context;

}









function attachExecutionMeta(

    result,

    context

) {


    return {


        ...result,


        executionMeta:

        {


            ...(result.executionMeta || {}),


            executionId:
                context.executionId,


            traceId:
                context.trace?.id || null,


            attempts:
                context.attempt,


            retryCount:
                context.retryCount || 0,


            replanCount:
                context.replanCount || 0,


            experience:

            {

                used:
                    context.experience?.found === true,


                skills:
                    context.experience?.skills || []

            }


        }


    };

}









function registerFailure(

    context,

    failure

) {


    if(
        !failure
    )
        return;



    if(
        !Array.isArray(
            context.errors
        )
    ){

        context.errors=[];

    }



    context.errors.push({

        ...failure,


        timestamp:

            new Date()
            .toISOString()

    });

}









async function executeReplan(

    context,

    decision

) {


    addTraceEvent(

        context.trace,

        "REPLAN_STARTED",

        {

            failure:
                decision.failure

        }

    );







    const previousPlan =
        context.plan;







    const alternative =

        await createAlternativePlan(

            context,

            decision.failure

        );





    if(
        !alternative?.success
    ){


        addTraceEvent(

            context.trace,

            "REPLAN_FAILED",

            alternative

        );


        return false;

    }







    const applied =

        applyAlternativePlan(

            context,

            alternative

        );





    if(
        !applied
    ){

        return false;

    }







    context.replanCount =

        Number(
            context.replanCount || 0
        )
        +
        1;






    context.retryCount = 0;








    addTraceReplan(

        context.trace,

        {


            previousPlan,


            newPlan:
                context.plan,


            failure:
                decision.failure


        }

    );





    addTraceEvent(

        context.trace,

        "REPLAN_COMPLETED"

    );



    return true;

}









export async function executePlanCycle(

    task,

    initialPlan,

    planningContext={}

) {


    const context =

        createJessicaContext({

            task,

            plan:
                initialPlan,

            planningContext

        });







    context.trace =

        createExecutionTrace(
            task
        );







    addTraceEvent(

        context.trace,

        "EXECUTION_STARTED",

        {

            executionId:
                context.executionId

        }

    );







    let lastFailure = null;








    while(

        context.attempt <
        MAX_EXECUTION_ATTEMPTS

    ){



        context.attempt++;






        addTraceAttempt(

            context.trace,

            context.attempt,

            {

                retryCount:
                    context.retryCount,


                replanCount:
                    context.replanCount


            }

        );







        let result;





        try {


            result =

                await executeExecutionStep(
                    context
                );


        } catch(error){


            result={


                success:false,


                failure:{

                    stage:
                        "execution",


                    failureType:
                        "execution-error",


                    reason:
                        error?.message ||

                        "Execution error"

                }

            };

        }








        updateTraceFromResult(

            context.trace,

            result

        );








        if(
            result?.success === true
        ){


            context.state =
                "COMPLETED";


            finishExecutionTrace(
                context.trace
            );


            return attachExecutionMeta(

                result.result,

                context

            );

        }








        lastFailure =

            result?.failure ||

            {

                stage:
                    "execution",


                failureType:
                    "unknown",


                reason:
                    "Unknown failure"

            };







        registerFailure(

            context,

            lastFailure

        );








        const decision =

            await handleExecutionFailure(

                context,

                lastFailure

            );








        switch(

            decision.action

        ){


            case FAILURE_ACTION.RETRY:


                context.retryCount =

                    Number(
                        context.retryCount || 0
                    )
                    +
                    1;



                addTraceEvent(

                    context.trace,

                    "RETRY"

                );


                continue;







            case FAILURE_ACTION.REPLAN:


                if(
                    await executeReplan(
                        context,
                        decision
                    )
                ){

                    continue;

                }


                break;







            case FAILURE_ACTION.CLARIFICATION:


                finishExecutionTrace(
                    context.trace
                );


                return attachExecutionMeta(

                    buildTerminalResult(

                        context,

                        {

                            needsClarification:
                                true,

                            ...lastFailure

                        }

                    ),

                    context

                );







            case FAILURE_ACTION.FINISH:

                break;


        }



        break;


    }








    finishExecutionTrace(

        context.trace

    );



    return attachExecutionMeta(

        buildTerminalResult(

            context,

            lastFailure ||

            {

                failureType:
                    "execution-limit",

                reason:
                    "Execution limit reached"

            }

        ),

        context

    );


}
