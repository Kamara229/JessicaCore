/*
 * =========================================================
 * JESSICA LEARNING PIPELINE v5
 * =========================================================
 *
 * Persistent Learning Pipeline.
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
 * затем независимо:
 *
 * learning_proposals
 *       ↓
 * PENDING_APPROVAL
 *       ↓
 * Runtime Proposal
 *       ↓
 * Approval Runner
 *
 *
 * ВАЖНО:
 *
 * Persistent Storage является
 * источником истины для Approval.
 *
 *
 * Поэтому:
 *
 * - рестарт процесса не теряет Proposal;
 * - старые PENDING Proposal восстанавливаются;
 * - Approval не зависит от Worker memory.
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
    getPendingLearningProposals
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
     * 1. RUN WORKER
     * =====================================================
     *
     * Worker failure не должен
     * автоматически блокировать
     * уже сохранённые Pending Proposals.
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
     * 2. READ PERSISTED PROPOSALS
     * =====================================================
     *
     * Это канонический источник
     * Proposal для Approval.
     *
     * =====================================================
     */


    let pendingResult;


    try {


        pendingResult =

            await getPendingLearningProposals();


    }catch(error){


        return {

            success:
                false,

            stage:
                "proposal-storage",

            worker:
                workerResult,

            workerError,

            proposals:
                [],

            error:

                error?.message

                ||

                "Pending Proposal Storage failed"

        };

    }


    if(
        pendingResult?.success !== true
    ){

        return {

            success:
                false,

            stage:
                "proposal-storage",

            worker:
                workerResult,

            workerError,

            proposals:
                [],

            error:

                pendingResult?.error

                ||

                "Не удалось получить Pending Learning Proposals"

        };

    }



    const proposals =

        Array.isArray(
            pendingResult.proposals
        )

            ? pendingResult.proposals

            : [];



    /*
     * =====================================================
     * 3. WORKER FAILED + NOTHING TO RECOVER
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
            "proposal",


        /*
         * Worker мог упасть,
         * но Recovery всё равно может
         * продолжить Approval старых Proposal.
         */


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
         * Сколько Proposal было произведено
         * Worker в текущем цикле.
         */


        created:

            countWorkerProposals(
                workerResult
            ),


        /*
         * Сколько Proposal реально ждут
         * Approval в persistent storage.
         */


        pending:

            proposals.length,


        proposals

    };

}
