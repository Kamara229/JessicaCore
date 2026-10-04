/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL STORAGE v5
 * =========================================================
 *
 * Public facade
 * Persistent Learning Proposal Storage.
 *
 *
 * proposalStorage/
 *
 * - Constants
 * - Utils
 * - Mapper
 * - Reader
 * - Writer
 * - Status
 * - Claim
 *
 * =========================================================
 */


export {

    saveLearningProposal,

    updateLearningProposalStatus,

    isSupportedLearningProposalStatus

} from "./proposalStorage/proposalStorageWriter.js";


export {

    findLearningProposalByQueueItemId,

    getPendingLearningProposals

} from "./proposalStorage/proposalStorageReader.js";


export {

    claimPendingLearningProposals

} from "./proposalStorage/proposalStorageClaim.js";
