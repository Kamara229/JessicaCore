import {PHONE_REPO} from "./phoneChatPolicy.js";
export const phoneRoot = `/repos/${PHONE_REPO}`;

export async function phoneGithub(path, {method = "GET", body} = {}) {
    const token = process.env.JESSICA_GITHUB_AUTOMATION_TOKEN;
    if (method !== "GET" && !token) throw Error("github-owner-setup-required");
    const response = await fetch("https://api.github.com" + phoneRoot + path, {
        method,
        headers: {Accept: "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
            ...(token ? {Authorization: `Bearer ${token}`} : {}),
            ...(body ? {"Content-Type": "application/json"} : {})},
        ...(body ? {body: JSON.stringify(body)} : {}),
        signal: AbortSignal.timeout(15000)
    });
    if (!response.ok) throw Error(`github-http-${response.status}`);
    return response.status === 204 ? null : response.json();
}

export async function publishPhoneCandidate(session, files, api = phoneGithub) {
    const branch = `phone-chat/${session.id}`;
    if (session.cancelled) throw Error("session-cancelled");
    const tree = await api("/git/trees", {method: "POST", body: {base_tree: session.treeSha, tree: files}});
    if (session.cancelled) throw Error("session-cancelled");
    const commit = await api("/git/commits", {method: "POST", body: {
        message: `Jessica ordinary-chat candidate: ${session.goal}`, tree: tree.sha, parents: [session.baseSha]
    }});
    if (session.cancelled) throw Error("session-cancelled");
    session.headSha = commit.sha;
    session.branch = branch;
    // Save the exact commit before the ref call: an ambiguous write is never rebuilt or resent.
    session.state = "publishing";
    await api("/git/refs", {method: "POST", body: {ref: `refs/heads/${branch}`, sha: commit.sha}});
    if (session.cancelled) throw Error("session-cancelled");
    session.state = "published";
    return openPhonePullRequest(session, api);
}

export async function openPhonePullRequest(session, api = phoneGithub) {
    if (session.cancelled) throw Error("session-cancelled");
    const existing = await api(`/pulls?state=all&head=Kamara229:${session.branch}`);
    if (session.cancelled) throw Error("session-cancelled");
    const pr = existing.find(item => item.head?.sha === session.headSha) || await api("/pulls", {
        method: "POST", body: {base: "main", head: session.branch,
            title: `[Jessica Phone] ${session.goal}`,
            body: "Source-only candidate from an owner-started ordinary ChatGPT session. Independent baseline/candidate tests are required. Chat text, account tokens and attachments are not published."}
    });
    session.pr = pr.number;
    session.url = pr.html_url;
    session.state = session.cancelled ? "cancelled" : "testing";
    return session;
}

export async function phoneVerification(session, api = phoneGithub) {
    const pr = await api(`/pulls/${session.pr}`);
    if (pr.head.sha !== session.headSha || pr.base.ref !== "main" || pr.head.repo?.full_name !== PHONE_REPO)
        throw Error("candidate-was-changed");
    if (pr.merged) return "merged";
    if (pr.state !== "open") return "closed";
    const workflow = await api("/actions/workflows/jessica-phone-verify.yml");
    const result = await api(`/actions/workflows/${workflow.id}/runs?event=pull_request&per_page=50`);
    const runs = result.workflow_runs.filter(run => run.event === "pull_request" &&
        run.head_sha === session.headSha && run.head_repository?.full_name === PHONE_REPO &&
        run.pull_requests?.some(item => item.number === session.pr));
    const latest = runs.sort((a, b) => b.id - a.id || (b.run_attempt || 1) - (a.run_attempt || 1))[0];
    if (!latest || latest.status !== "completed") return "testing";
    return latest.conclusion === "success" ? "verified" : "tests-failed";
}
