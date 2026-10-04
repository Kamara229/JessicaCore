/*
 * =========================================================
 * JESSICA LEARNING PIPELINE v7
 * =========================================================
 *
 * Persistent + Recoverable +
 * Atomic Learning Pipeline.
 *
 *
 * Flow:
 *
 * Learning Queue
 *       ↓
 * Worker
 *       ↓
 * Proposal Storage
 *
 *
 * PROCESSING stale
 *       ↓
 * Recovery
 *       ↓
 * PENDING_APPROVAL
 *
 *
 * PENDING_APPROVAL
 *       ↓
 * Atomic Claim
 *       ↓
 * PROCESSING
 *       ↓
 * Approval Runner
 *
 *
 * Persistent Storage является
 * источником истины.
 *
 *
 * НЕ:
 *
 * - принимает Learning Decision;
 * - создаёт Skill;
 * - сохраняет Experience.
 *
 * =========================================================
 */


import {
    runLearningWorker
} from "./learningWorker.js";


import {
    recoverStaleLearningProposals,
    claimPendingLearningProposals
} from "./learningProposalStorage.js";





/*
 * =========================================================
 * WORKER PROPOSAL COUNT
 * =========================================================
 */


function countWorkerProposals(
    workerResult
) {

    if(
        !Array.isArray(
            workerResult?.results
        )
    ){

        return 0;

    }


    return workerResult.results

        .filter(

            item =>

                item?.success === true

                &&

                item?.proposalId

        )

        .length;

}





/*
 * =========================================================
 * WORKER STATS
 * =========================================================
 */


function buildWorkerStats(
    workerResult
) {

    const results =

        Array.isArray(
            workerResult?.results
        )

            ? workerResult.results

            : [];


    return {

        processed:

            Number(
                workerResult?.processed
            )

            ||

            results.length,


        successful:

            Number(
                workerResult?.successful
            )

            ||

            results.filter(

                item =>
                    item?.success === true

            )
            .length,


        ignored:

            Number(
                workerResult?.ignored
            )

            ||

            results.filter(

                item =>
                    item?.ignored === true

            )
            .length,


        failed:

            Number(
                workerResult?.failed
            )

            ||

            results.filter(

                item =>
                    item?.success === false

            )
            .length

    };

}





/*
 * =========================================================
 * RUN PIPELINE
 * =========================================================
 */


export async function runLearningPipeline()
{


    /*
     * =====================================================
     * 1. PRODUCE NEW PROPOSALS
     * =====================================================
     */


    let workerResult = null;

    let workerError = null;


    try {


        workerResult =

            await runLearningWorker();


        if(
            workerResult?.success !== true
        ){

            workerError =

                workerResult?.error

                ||

                "Learning Worker failed";

        }


    }catch(error){


        workerError =

            error?.message

            ||

            "Learning Worker failed";

    }



    /*
     * =====================================================
     * 2. RECOVER STALE PROCESSING
     * =====================================================
     *
     * Recovery failure не должен
     * блокировать обычные новые Proposal.
     *
     * Он фиксируется отдельно.
     *
     * =====================================================
     */


    let recoveryResult;


    try {


        recoveryResult =

            await recoverStaleLearningProposals();


    }catch(error){


        recoveryResult = {

            success:
                false,

            recovered:
                0,

            error:

                error?.message

                ||

                "Learning Proposal recovery failed"

        };

    }


    if(
        recoveryResult?.success !== true
    ){

        console.error(

            "Jessica Learning Pipeline recovery warning:",

            recoveryResult?.error

            ||

            "unknown recovery error"

        );

    }



    /*
     * =====================================================
     * 3. ATOMIC CLAIM
     * =====================================================
     */


    let claimResult;


    try {


        claimResult =

            await claimPendingLearningProposals();


    }catch(error){


        return {

            success:
                false,

            stage:
                "proposal-claim",

            worker:
                workerResult,

            workerError,

            recovery:
                recoveryResult,

            proposals:
                [],

            error:

                error?.message

                ||

                "Learning Proposal claim failed"

        };

    }


    if(
        claimResult?.success !== true
    ){

        return {

            success:
                false,

            stage:
                "proposal-claim",

            worker:
                workerResult,

            workerError,

            recovery:
                recoveryResult,

            proposals:
                [],

            error:

                claimResult?.error

                ||

                "Не удалось atomically claim Learning Proposals"

        };

    }



    const proposals =

        Array.isArray(
            claimResult.proposals
        )

            ? claimResult.proposals

            : [];



    /*
     * =====================================================
     * 4. WORKER FAILED + NOTHING CLAIMED
     * =====================================================
     */


    if(
        workerError
        &&
        proposals.length === 0
    ){

        return {

            success:
                false,

            stage:
                "worker",

            worker:
                workerResult,

            workerError,

            recovery:
                recoveryResult,

            proposals:
                [],

            error:
                workerError

        };

    }



    /*
     * =====================================================
     * 5. RESULT
     * =====================================================
     */


    return {

        success:
            true,


        stage:
            "proposal-claimed",


        workerSuccess:

            workerError === null,


        workerError,


        stats:

            buildWorkerStats(
                workerResult
            ),


        worker:
            workerResult,


        created:

            countWorkerProposals(
                workerResult
            ),


        /*
         * Crash Recovery.
         */


        recovered:

            Number(
                recoveryResult?.recovered || 0
            ),


        recovery:

            recoveryResult,


        /*
         * Эксклюзивно захвачены
         * данным Daemon Cycle.
         */


        claimed:

            proposals.length,


        proposals

    };

}
