/**
 * Privacy-preserving diagnostic phases for the owner-started Android relay.
 * No conversation text, code, prompt, result, device ID or credentials.
 */
export const CODEX_PHASES = Object.freeze(new Set([
    "started","chat_selected","draft_ready","draft_unverified","draft_retry","send_requested","sent",
    "reply_streaming","awaiting_markers","reply_stabilizing",
    "reply_partial","reply_out_of_view","reply_detected","reply_seeking_start",
    "reply_verified","interpret_started","interpret_retry","interpret_failed","next_ready",
    "chat_away","chat_resumed","conversation_length_notice","window_snapshot_retry","editor_missing","send_control_missing","send_uncertain","stopped"
]));
const records = new Map();
const SESSION = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function validateCodexPhase(input) {
    if (!input || typeof input !== "object" || Array.isArray(input) ||
        !SESSION.test(input.session) || !CODEX_PHASES.has(input.phase) ||
        !Number.isInteger(input.turn) || input.turn < 0 || input.turn > 60 ||
        !Number.isInteger(input.elapsedSeconds) ||
        input.elapsedSeconds < 0 || input.elapsedSeconds > 30_000) {
        throw Error("invalid-relay-phase");
    }
    return {session: input.session, phase: input.phase, turn: input.turn,
        elapsedSeconds: input.elapsedSeconds};
}

export function recordCodexPhase(input, logger=console.info) {
    const item = validateCodexPhase(input);
    const now = Date.now();
    // A bounded, non-persistent rate limit; server restarts reset it.
    for (const [key, info] of records) if (now - info.start > 9 * 60 * 60_000) records.delete(key);
    const info = records.get(item.session) || {start:now,count:0};
    if (info.count >= 120) return false;
    info.count++;
    records.set(item.session,info);
    logger("Jessica Codex relay phase:", item);
    return true;
}
