import {PHONE_GOALS, phoneId, buildPhonePrompt, validatePhonePatch} from "./phoneChatPolicy.js";
import {phoneGithub, publishPhoneCandidate, openPhonePullRequest, phoneVerification} from "./phoneChatGithub.js";

// Ephemeral capabilities: a Render restart pauses the phone; it never silently starts another job.
const sessions = new Map();
const TTL = 90 * 60 * 1000;
function prune() { for (const [key, value] of sessions) if (Date.now() > value.expires) sessions.delete(key); }
function get(id) {
    prune();
    const session = sessions.get(id);
    if (!session) throw Error("session-expired-start-again");
    return session;
}
const publicState = s => ({id: s.id, state: s.state, pr: s.pr || null, url: s.url || null,
    autoApply: s.autoApply, usesPaidModel: false});

export async function preparePhoneChat(goal, autoApply = false, api = phoneGithub) {
    if (!Object.hasOwn(PHONE_GOALS, goal)) throw Error("invalid-goal");
    prune();
    if (sessions.size >= 10) throw Error("too-many-active-sessions");
    const head = await api("/git/ref/heads/main");
    const baseSha = head.object.sha;
    const commit = await api(`/git/commits/${baseSha}`);
    const sources = [];
    for (const path of PHONE_GOALS[goal]) {
        const blob = await api(`/contents/${path}?ref=${baseSha}`);
        if (blob.type !== "file" || blob.encoding !== "base64" || blob.size > 14000) throw Error("source-unavailable-or-too-large");
        sources.push({path, sha: blob.sha, content: Buffer.from(blob.content, "base64").toString("utf8")});
    }
    const session = {id: phoneId(), baseSha, treeSha: commit.tree.sha, sources, goal,
        autoApply: autoApply === true, expires: Date.now() + TTL, state: "prepared", busy: false};
    sessions.set(session.id, session);
    return {...publicState(session), baseSha, prompt: buildPhonePrompt(session)};
}

export async function submitPhoneChat(id, reply, api = phoneGithub) {
    const session = get(id);
    if (session.cancelled) throw Error("session-cancelled");
    if (session.busy) throw Error("session-busy");
    if (["testing", "verified", "merged", "no-change"].includes(session.state)) return publicState(session);
    if (!["prepared", "published"].includes(session.state)) throw Error("submission-status-unknown-check-github");
    session.busy = true;
    try {
        if (session.state === "published") {
            await openPhonePullRequest(session, api);
            return publicState(session);
        }
        const files = validatePhonePatch(reply, session);
        if (!files.length) { session.state = "no-change"; return publicState(session); }
        const head = await api("/git/ref/heads/main");
        if (head.object.sha !== session.baseSha) throw Error("main-changed-start-again");
        if (session.cancelled) throw Error("session-cancelled");
        await publishPhoneCandidate(session, files, api);
        return publicState(session);
    } finally { session.busy = false; }
}

export async function readPhoneChat(id, api = phoneGithub) {
    const session = get(id);
    if (session.busy || !["testing", "verified"].includes(session.state)) return publicState(session);
    session.busy = true;
    try {
        const verified = await phoneVerification(session, api);
        if (session.cancelled) return publicState(session);
        session.state = verified;
        if (verified === "verified" && session.autoApply) {
            const head = await api("/git/ref/heads/main");
            if (session.cancelled) return publicState(session);
            if (head.object.sha !== session.baseSha) { session.state = "stale-base"; return publicState(session); }
            session.state = "applying";
            // The tested commit has exactly baseSha as parent. A non-forced ref update
            // atomically fails if main advanced, unlike a PR merge that could combine an untested new base.
            const result = await api("/git/refs/heads/main", {method: "PATCH", body: {
                sha: session.headSha, force: false
            }});
            session.state = result.object?.sha === session.headSha ? "merged" : "application-status-unknown";
        }
        return publicState(session);
    } catch (error) {
        if (session.state === "applying") session.state = "application-status-unknown";
        throw error;
    } finally { session.busy = false; }
}

export function stopPhoneChat(id) {
    const session = get(id);
    session.cancelled = true;
    if (!["merged", "applying", "application-status-unknown"].includes(session.state)) session.state = "cancelled";
    return publicState(session);
}
