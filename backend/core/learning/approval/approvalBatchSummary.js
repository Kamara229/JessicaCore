/*
 * =========================================================
 * JESSICA APPROVAL BATCH SUMMARY
 * =========================================================
 */


import {
    APPROVAL_RESULT_STATUS
} from "./approvalConstants.js";


export function summarizeApprovalResults(
    proposals,
    results
) {

    return {

        success:
            true,

        processed:

            Array.isArray(proposals)

                ? proposals.length

                : 0,

        learned:

            results.filter(
                item =>
                    item.learned === true
            )
            .length,

        reinforced:

            results.filter(
                item =>
                    item.reinforced === true
            )
            .length,

        candidates:

            results.filter(
                item =>
                    item.status ===
                    APPROVAL_RESULT_STATUS.KEEP_CANDIDATE
            )
            .length,

        rejected:

            results.filter(
                item =>
                    item.status ===
                    APPROVAL_RESULT_STATUS.REJECTED
            )
            .length,

        failed:

            results.filter(
                item =>
                    item.status ===
                    APPROVAL_RESULT_STATUS.FAILED
            )
            .length,

        candidateMemoryErrors:

            results.filter(
                item =>

                    item.learned === true

                    &&

                    item.candidateMemoryStateUpdated === false

                    &&

                    item
                        ?.candidateMemory
                        ?.skipped !== true
            )
            .length,

        results

    };

}
