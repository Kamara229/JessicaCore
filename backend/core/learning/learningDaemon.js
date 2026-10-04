/*
 * =========================================================
 * JESSICA LEARNING DAEMON v6
 * =========================================================
 *
 * Фоновый процесс
 * автономного обучения Jessica.
 *
 *
 * Flow:
 *
 * Timer
 *      ↓
 * Learning Pipeline
 *      ↓
 * Persisted PENDING Proposals
 *      ↓
 * Approval Runner
 *      ↓
 * Reviewer
 *      ↓
 * Quality Gate
 *      ↓
 * Autonomy
 *      ↓
 * Experience / Candidate Memory
 *
 *
 * ВАЖНО:
 *
 * Proposal поступают в Approval
 * из persistent storage,
 * а не напрямую из Worker Memory.
 *
 *
 * НЕ:
 *
 * - анализирует Execution;
 * - создаёт Proposal;
 * - создаёт Skill;
 * - работает с Supabase напрямую.
 *
 * =========================================================
 */


import {
    runLearningPipeline
} from "./learningPipeline.js";


import {
    runLearningApproval
} from "./learningApprovalRunner.js";





/*
 * =========================================================
 * STATE
 * =========================================================
 */


let daemonTimer =
    null;


let running =
    false;





/*
 * =========================================================
 * NUMBER
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
 * EMPTY RESULT
 * =========================================================
 */


function createEmptyCycleResult(
    reason,
    pipeline = null
) {

    return {

        success:
            true,


        reason:

            reason

            ||

            "nothing-to-learn",


        pipeline,


        created:

            normalizeNumber(
                pipeline?.created
            ),


        pending:
            0,


        proposals:
            0,


        learned:
            0,


        candidates:
            0,


        rejected:
            0,


        failed:
            0,


        candidateMemoryErrors:
            0,


        approval:
            null

    };

}





/*
 * =========================================================
 * SINGLE LEARNING CYCLE
 * =========================================================
 */


export async function runLearningDaemonCycle()
{

    /*
     * =====================================================
     * LOCAL PROCESS LOCK
     * =====================================================
     *
     * Защищает один Node process
     * от overlapping timer cycles.
     *
     * Distributed locking для нескольких
     * Render instances — отдельный этап.
     *
     * =====================================================
     */


    if(
        running
    ){

        return {

            success:
                false,

            reason:
                "Learning cycle already running"

        };

    }


    running =
        true;


    try {


        /*
         * =================================================
         * 1. LEARNING PIPELINE
         * =================================================
         */


        const pipelineResult =

            await runLearningPipeline();


        if(
            !pipelineResult?.success
        ){

            return {

                success:
                    false,

                stage:

                    pipelineResult?.stage

                    ||

                    "pipeline",

                pipeline:
                    pipelineResult,

                error:

                    pipelineResult?.error

                    ||

                    "Learning Pipeline failed"

            };

        }



        /*
         * =================================================
         * 2. PERSISTED PROPOSALS
         * =================================================
         */


        const proposals =

            Array.isArray(
                pipelineResult.proposals
            )

                ? pipelineResult.proposals

                : [];


        if(
            proposals.length === 0
        ){

            return createEmptyCycleResult(

                "no-pending-proposals",

                pipelineResult

            );

        }



        /*
         * =================================================
         * 3. APPROVAL RUNNER
         * =================================================
         */


        let approvalResult;


        try {


            approvalResult =

                await runLearningApproval(
                    proposals
                );


        }catch(error){


            return {

                success:
                    false,

                stage:
                    "approval",

                pipeline:
                    pipelineResult,

                proposals:
                    proposals.length,

                error:

                    error?.message

                    ||

                    "Learning Approval failed"

            };

        }



        /*
         * Approval Runner сам должен
         * вернуть корректный batch result.
         */


        if(
            approvalResult?.success !== true
        ){

            return {

                success:
                    false,

                stage:
                    "approval",

                pipeline:
                    pipelineResult,

                proposals:
                    proposals.length,

                approval:
                    approvalResult,

                error:

                    approvalResult?.error

                    ||

                    "Learning Approval Runner failed"

            };

        }



        /*
         * =================================================
         * 4. FINAL RESULT
         * =================================================
         */


        return {

            success:
                true,


            pipeline:
                pipelineResult,


            /*
             * Worker production
             */


            created:

                normalizeNumber(
                    pipelineResult.created
                ),


            /*
             * Persistent Proposal workload
             */


            pending:

                proposals.length,


            proposals:

                proposals.length,


            /*
             * Approval outcomes
             */


            learned:

                normalizeNumber(
                    approvalResult.learned
                ),


            candidates:

                normalizeNumber(
                    approvalResult.candidates
                ),


            rejected:

                normalizeNumber(
                    approvalResult.rejected
                ),


            failed:

                normalizeNumber(
                    approvalResult.failed
                ),


            candidateMemoryErrors:

                normalizeNumber(
                    approvalResult.candidateMemoryErrors
                ),


            /*
             * Worker мог иметь ошибку,
             * но Recovery старых Proposal
             * всё равно мог выполниться.
             */


            workerSuccess:

                pipelineResult.workerSuccess !== false,


            workerError:

                pipelineResult.workerError

                ||

                null,


            approval:
                approvalResult

        };


    }catch(error){


        console.error(

            "Jessica Learning Daemon error:",

            error

        );


        return {

            success:
                false,

            error:

                error?.message

                ||

                "Learning Daemon failed"

        };


    }finally{


        running =
            false;

    }

}





/*
 * =========================================================
 * START DAEMON
 * =========================================================
 */


export function startLearningDaemon(
    interval = 15 * 60 * 1000
) {

    if(
        daemonTimer
    ){

        return {

            started:
                false,

            reason:
                "Daemon already started"

        };

    }


    daemonTimer =

        setInterval(

            () => {


                runLearningDaemonCycle()

                    .catch(

                        error => {


                            console.error(

                                "Learning Daemon cycle error:",

                                error

                            );

                        }

                    );

            },

            interval

        );


    console.log(

        "Jessica Learning Daemon started"

    );


    return {

        started:
            true,

        interval

    };

}





/*
 * =========================================================
 * STOP DAEMON
 * =========================================================
 */


export function stopLearningDaemon()
{

    if(
        !daemonTimer
    ){

        return {

            stopped:
                false

        };

    }


    clearInterval(
        daemonTimer
    );


    daemonTimer =
        null;


    return {

        stopped:
            true

    };

}
