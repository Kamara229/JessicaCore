/**
 * Per-process quota circuit for the Codex interpretation endpoint only.
 * A 429 should not trigger another doomed Groq call on every relay turn.
 * No keys, prompts, user messages or test results are stored.
 */
const COOLDOWN_MS = 10 * 60 * 1000;
let blockedUntil = 0;

export function isCodexProviderQuota(error) {
    const status = error?.status ?? error?.response?.status;
    const detail = String(error?.message || error || "");
    return status === 429 ||
        /(?:\b429\b|rate[_ -]?limit|quota|tokens? per day|resources?[_ -]?exhausted)/iu.test(detail);
}

export function recordCodexQuota(error, now = Date.now()) {
    if (isCodexProviderQuota(error))
        blockedUntil = Math.max(blockedUntil, now + COOLDOWN_MS);
    return blockedUntil > now;
}

export function isCodexQuotaOpen(now = Date.now()) {
    return now < blockedUntil;
}
