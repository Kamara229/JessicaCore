/** Resolve persisted state from the actual version on idempotent saves.
 * Legacy RPC v2 may label an existing draft/disabled Skill as "published"
 * even though the immutable payload and current row say otherwise.
 */
export function resolveStoredExperienceStatus(rpcResult) {
    const existing=rpcResult?.existing===true||rpcResult?.existing==="true";
    const state=existing?rpcResult?.experience?.status:rpcResult?.status;
    return typeof state==="string"&&state.trim()?state.trim():null;
}
