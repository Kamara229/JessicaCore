/*
 * =========================================================
 * JESSICA LEARNING PIPELINE v8
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
import {recoverStaleLearningQueue} from "./learningQueueRecovery.js";
import { consolidateLearningCandidates } from "./candidateMemory/candidateConsolidator.js";


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


        deferred:

            Number(
                workerResult?.deferred
            )

            ||

            results.filter(

                item =>
                    item?.deferred === true

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


    // Quarantine ambiguous stale Candidate Memory actions before normal claims.
    let queueRecovery;
    try {
        queueRecovery=await recoverStaleLearningQueue();
    } catch (error) {
        queueRecovery={success:false,error:error?.message||"Queue recovery error"};
    }
    if(queueRecovery?.success!==true){
        console.warn("Jessica stale Learning Queue recovery:",queueRecovery?.error);
    }

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
     * 2.5. Autonomous reconciliation of earlier duplicate ACTIVE candidates.
     * This doesn't publish a Skill; quality and autonomy gates remain the
     * only routes to published Experience.
     */
    let consolidation = null;
    try {
        consolidation = await consolidateLearningCandidates();
    } catch (error) {
        // A temporary database error must not block the existing learning cycle.
        consolidation = {
            success: false, considered: 0, merged: 0,
            error: error?.message || "Candidate consolidation failed"
        };
        console.warn("Jessica Candidate consolidation warning:", consolidation.error);
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

        queueRecovery,

        created:

            countWorkerProposals(
                workerResult
            ),


        /*
         * Crash Recovery.
         */


        consolidation,

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
