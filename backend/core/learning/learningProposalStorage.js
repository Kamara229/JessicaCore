/*
 * =========================================================
 * JESSICA LEARNING PROPOSAL STORAGE v4
 * =========================================================
 *
 * Public facade
 * Persistent Learning Proposal Storage.
 *
 *
 * Внутренняя реализация разделена:
 *
 * proposalStorage/
 *
 * - Constants
 * - Utils
 * - Mapper
 * - Reader
 * - Status
 * - Writer
 *
 *
 * Благодаря этому остальные части Jessica
 * продолжают импортировать:
 *
 * ./learningProposalStorage.js
 *
 * и не знают внутреннюю структуру Storage.
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
