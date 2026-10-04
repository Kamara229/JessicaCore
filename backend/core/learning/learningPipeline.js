/*
 * =========================================================
 * JESSICA LEARNING PIPELINE v6
 * =========================================================
 *
 * Persistent + Atomic Learning Pipeline.
 *
 *
 * Flow:
 *
 * Learning Queue
 *       ↓
 * Learning Worker
 *       ↓
 * Proposal Storage
 *
 *
 * затем:
 *
 * PENDING_APPROVAL
 *       ↓
 * ATOMIC CLAIM
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
 * Atomic Claim гарантирует:
 *
 * один Proposal
 *      ↓
 * один одновременно работающий
 * Approval Cycle
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
     *
     * Worker failure не должен
     * блокировать Approval уже сохранённых
     * Proposal.
     *
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
     * 2. ATOMIC CLAIM
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
     * 3. WORKER FAILED + NOTHING CLAIMED
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

            proposals:
                [],

            error:
                workerError

        };

    }



    /*
     * =====================================================
     * 4. RESULT
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


        /*
         * Proposal произведены
         * текущим Worker cycle.
         */


        created:

            countWorkerProposals(
                workerResult
            ),


        /*
         * Proposal успешно и эксклюзивно
         * забраны этим Daemon cycle.
         */


        claimed:

            proposals.length,


        proposals

    };

}
