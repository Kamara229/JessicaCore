import { candidateProcedureFamily } from "./candidateFamily.js";
import { calculateCandidateSimilarity } from "./candidateMatcher.js";
import { mergeLearningCandidates } from "./candidateMerge.js";
import {
    listActiveCandidatesByCategory,
    updateCandidateMemory,
    updateCandidateMemoryStatus
} from "./candidateStorage.js";

const MIN_SIMILARITY = 0.75;
const MAX_PER_CYCLE = 8;

/**
 * Pure, conservative consolidation plan. It doesn't need human approval:
 * only ACTIVE candidates of one recognized, verified procedural family match.
 * Keep the oldest candidate as the stable skill identity. Never conflate
 * unrelated families, or re-publish a Skill on consolidation alone.
 */
export function planCandidateConsolidation(memories) {
    if (!Array.isArray(memories)) return [];
    const active = memories
        .filter(m => m && m.id && m.status === "ACTIVE" && m.candidate)
        .slice()
        .sort((a, b) => String(a.createdAt || "").localeCompare(String(b.createdAt || "")));
    const plan = [];
    const claimed = new Set();

    for (let i = 0; i < active.length; i++) {
        const canonical = active[i];
        if (claimed.has(canonical.id)) continue;
        const family = candidateProcedureFamily(canonical.candidate);
        if (!family) continue;

        for (let j = i + 1; j < active.length; j++) {
            const duplicate = active[j];
            if (claimed.has(duplicate.id)) continue;
            if (candidateProcedureFamily(duplicate.candidate) !== family) continue;

            const similarity = calculateCandidateSimilarity(canonical.candidate, duplicate.candidate);
            if (similarity < MIN_SIMILARITY) continue;

            plan.push({
                canonical,
                duplicate,
                family,
                similarity
            });
            claimed.add(duplicate.id);
            claimed.add(canonical.id); // One update per canonical in each cycle.
            if (plan.length >= MAX_PER_CYCLE) return plan;
            break;
        }
    }
    return plan;
}

/**
 * Called by the Learning Daemon, not by a person. Consolidation aggregates
 * independently verified observations; it never promotes or approves Skills.
 * On failure, leave the duplicate ACTIVE so a later cycle can safely retry.
 */
export async function consolidateLearningCandidates() {
    const active = await listActiveCandidatesByCategory(null, 50);
    const plan = planCandidateConsolidation(active);
    let mergedCount = 0;
    let skipped = 0;
    let failed = 0;

    for (const item of plan) {
        const {canonical, duplicate, similarity, family} = item;
        try {
            const merged = mergeLearningCandidates({
                existing: canonical.candidate,
                incoming: duplicate.candidate,
                exactMatch: false,
                similarity
            });

            const oldKeys = new Set((canonical.candidate.examples || [])
                .map(x => x?.traceId || x?.result?.executionMeta?.traceId)
                .filter(Boolean));
            const newKeys = new Set((merged.examples || [])
                .map(x => x?.traceId || x?.result?.executionMeta?.traceId)
                .filter(Boolean));

            // A prior cycle may have persisted the canonical merge but failed
            // to retire the duplicate (e.g. before a DB constraint migration).
            // Retrying must finish retirement WITHOUT inflating evidence twice.
            const alreadyRecorded =
                Array.isArray(canonical.metadata?.consolidatedFrom) &&
                canonical.metadata.consolidatedFrom.includes(duplicate.id);
            const addsIndependentTrace = newKeys.size > oldKeys.size;
            if (!addsIndependentTrace && !alreadyRecorded) {
                skipped++;
                continue;
            }

            if (addsIndependentTrace && !alreadyRecorded) {
                await updateCandidateMemory({
                id: canonical.id,
                expectedUpdatedAt:canonical.updatedAt,
                expectedRevision:canonical.revision,
                candidate: merged,
                skillId: merged.skillId,
                metadata: {
                    ...canonical.metadata,
                    lastMatchType: "AUTONOMOUS_CONSOLIDATION",
                    lastSimilarity: similarity,
                    family,
                    consolidatedFrom: [
                        ...(Array.isArray(canonical.metadata?.consolidatedFrom)
                            ? canonical.metadata.consolidatedFrom : []),
                        duplicate.id
                    ]
                }
            });
            }

            const retired = await updateCandidateMemoryStatus({
                id: duplicate.id,
                status: "MERGED",
                metadata: {
                    mergedIntoId: canonical.id,
                    mergedIntoSkillId: merged.skillId,
                    mergedAt: new Date().toISOString(),
                    mergeFamily: family
                }
            });
            if (retired?.success !== true) {
                failed++;
                // The canonical update is idempotent by trace ID; retry next cycle.
                console.warn("Jessica Candidate consolidation could not retire duplicate:", {
                    family, reason: retired?.error || "unknown"
                });
            } else {
                mergedCount++;
            }
        } catch (error) {
            failed++;
            console.warn("Jessica Candidate consolidation error:", {
                family, message: error?.message || "unknown"
            });
        }
    }

    if (plan.length) console.log("Jessica Candidate consolidation:", {
        considered: plan.length, merged: mergedCount, skipped, failed
    });

    return {success: failed === 0, considered: plan.length, merged: mergedCount, skipped, failed};
}
