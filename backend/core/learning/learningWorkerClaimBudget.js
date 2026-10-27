// Claim only the work one sequential Worker cycle can safely finish.
// Pattern extraction has a five-minute hard timeout. Eight such tasks
// take at most 40 minutes, well below the 120-minute stale-lease floor.
// A 100-item claim could leave later items idle until their leases expire.
export const LEARNING_WORKER_CLAIM_LIMIT=8;
