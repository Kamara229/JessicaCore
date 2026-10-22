/** Privacy-safe progress events. No ChatGPT text, titles, URLs or device IDs. */
export const RELAY_PHASES = new Set([
    "chat-selected","message-prepared","message-sent","waiting-reply",
    "reply-verified","decision-running","next-message-ready",
    "composer-missing","chat-not-visible","reply-timeout","stopped"
]);

export function validateRelayProgress(body) {
    const session = typeof body?.session === "string" ? body.session : "";
    if (!/^[a-f0-9-]{36}$/i.test(session)) throw Error("invalid-session");
    const phase = body?.phase;
    if (!RELAY_PHASES.has(phase)) throw Error("invalid-phase");
    const turn = Number(body?.turn);
    if (!Number.isInteger(turn) || turn < 0 || turn > 60) throw Error("invalid-turn");
    const count = key => {
        const v = Number(body?.[key]);
        if (!Number.isInteger(v) || v < 0 || v > 1500) throw Error("invalid-count");
        return v;
    };
    return {
        session, phase, turn,
        sawReplyStart: body?.sawReplyStart === true,
        sawReplyEnd: body?.sawReplyEnd === true,
        generating: body?.generating === true,
        composerCandidates: count("composerCandidates"),
        sendCandidates: count("sendCandidates")
    };
}
