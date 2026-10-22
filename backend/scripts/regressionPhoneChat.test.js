import test from "node:test";
import assert from "node:assert/strict";
import {validatePhonePatch, phoneTestPath, buildPhonePrompt} from "../core/codex/phoneChatPolicy.js";
import {preparePhoneChat, submitPhoneChat, readPhoneChat, stopPhoneChat} from "../core/codex/phoneChatService.js";
import {phoneVerification} from "../core/codex/phoneChatGithub.js";

const source = {path: "backend/core/chat/conversationContext.js", sha: "b".repeat(40), content: "export const x=1;"};
const session = {id: "12345678-abcd-4abc-8abc-123456789012", baseSha: "a".repeat(40), sources: [source]};
const reply = () => ({nonce: session.id, baseSha: session.baseSha, status: "change", files: [
    {...source, content: "export const x=2;"},
    {path: phoneTestPath(session.id), sha: null, content: "import test from 'node:test';"}
]});
test("candidate is tied to exact session and source; protected paths and unchanged files are rejected", () => {
    assert.equal(validatePhonePatch(reply(), session).length, 2);
    for (const mutate of [r => r.nonce = "other", r => r.baseSha = "other",
        r => r.files[0].sha = "old", r => r.files[0].path = "../../server.js",
        r => r.files[0].path = ".github/workflows/android.yml", r => r.files[0].content = source.content,
        r => r.files.push(r.files[0]), r => r.files.pop(), r => r.files[1].sha = "existing",
        r => r.files[0].content = "x".repeat(33000)]) {
        const value = reply(); mutate(value); assert.throws(() => validatePhonePatch(value, session));
    }
    assert.deepEqual(validatePhonePatch({...reply(), status: "no_change", files: []}, session), []);
    assert.throws(() => validatePhonePatch({...reply(), status: "no_change"}, session));
});
test("prompt contains bounded source context and no standalone reply envelope to confuse with assistant output", () => {
    const prompt = buildPhonePrompt(session);
    assert.ok(prompt.includes(source.content));
    assert.ok(!prompt.split("\n").includes("JESSICA_REPLY_" + session.id));
    assert.ok(!prompt.includes("ownerKey"));
});
const prepareApi = async path => {
    if (path === "/git/ref/heads/main") return {object: {sha: session.baseSha}};
    if (path.startsWith("/git/commits/")) return {tree: {sha: "c".repeat(40)}};
    return {type: "file", encoding: "base64", size: 17, sha: source.sha,
        content: Buffer.from(source.content).toString("base64")};
};
test("no-change never publishes or claims verified learning; cancellation blocks submission", async () => {
    const prepared = await preparePhoneChat("context", false, prepareApi);
    let writes = 0;
    const noChange = await submitPhoneChat(prepared.id, {nonce: prepared.id, baseSha: prepared.baseSha,
        status: "no_change", files: []}, async () => { writes++; });
    assert.equal(noChange.state, "no-change"); assert.equal(writes, 0);
    const cancelled = await preparePhoneChat("language", false, prepareApi);
    stopPhoneChat(cancelled.id);
    await assert.rejects(submitPhoneChat(cancelled.id, {}, async () => { writes++; }), /cancelled/);
    assert.equal(writes, 0);
});
test("verification cannot use a different head, foreign repository, unrelated PR or old successful run", async () => {
    const current = {...session, headSha: "d".repeat(40), pr: 123};
    const pr = {head: {sha: current.headSha, repo: {full_name: "Kamara229/JessicaCore"}}, base: {ref: "main"}, state: "open"};
    const success = {id: 1, head_sha: current.headSha, event: "pull_request", status: "completed", conclusion: "success",
        head_repository: {full_name: "Kamara229/JessicaCore"}, pull_requests: [{number: 123}]};
    const api = runs => async path => path.startsWith("/pulls/") ? pr : path.includes("/runs?") ?
        {workflow_runs: runs} : {id: 789};
    assert.equal(await phoneVerification(current, api([success])), "verified");
    assert.equal(await phoneVerification(current, api([{...success, head_sha: "e".repeat(40)}])), "testing");
    assert.equal(await phoneVerification(current, api([{...success, pull_requests: [{number: 124}]}])), "testing");
    assert.equal(await phoneVerification(current, api([success, {...success, id: 2, conclusion: "failure"}])), "tests-failed");
    await assert.rejects(phoneVerification(current, async () => ({...pr, head: {...pr.head, sha: "other"}})), /changed/);
});
test("a stopped session cannot merge even if a previously requested status later reports success", async () => {
    const prepared = await preparePhoneChat("context", true, prepareApi);
    let prCreated = false;
    const candidate = {...reply(), nonce: prepared.id, baseSha: prepared.baseSha,
        files: [{...source, content: "export const x=2;"}, {path: phoneTestPath(prepared.id), sha: null, content: "test"}]};
    const api = async (path, args) => {
        if (path === "/git/ref/heads/main") return {object: {sha: session.baseSha}};
        if (path === "/git/trees") return {sha: "tree"};
        if (path === "/git/commits") return {sha: "head"};
        if (path === "/git/refs") return {};
        if (path.startsWith("/pulls?")) return [];
        if (path === "/pulls" && args.method === "POST") { prCreated = true; return {number: 321, html_url: "https://github.com/Kamara229/JessicaCore/pull/321"}; }
        if (path === "/pulls/321") return {head: {sha: "head", repo: {full_name: "Kamara229/JessicaCore"}}, base: {ref: "main"}, state: "open"};
        if (path.endsWith(".yml")) return {id: 1};
        if (path.includes("/runs?")) { stopPhoneChat(prepared.id); return {workflow_runs: [{id: 1, event: "pull_request", head_sha: "head", head_repository: {full_name: "Kamara229/JessicaCore"}, pull_requests: [{number: 321}], status: "completed", conclusion: "success"}]}; }
        throw Error("Unexpected call, including forbidden merge: " + path);
    };
    await submitPhoneChat(prepared.id, candidate, api);
    assert.ok(prCreated);
    const stopped = await readPhoneChat(prepared.id, api);
    assert.equal(stopped.state, "cancelled");
});

test("automatic application uses a non-forced fast-forward and remains opt-in", async () => {
    for (const autoApply of [false, true]) {
        const prepared = await preparePhoneChat("context", autoApply, prepareApi);
        let applied = 0;
        const headSha = "d".repeat(40);
        const api = async (path, args) => {
            if (path === "/git/ref/heads/main") return {object: {sha: session.baseSha}};
            if (path === "/git/trees") return {sha: "tree"};
            if (path === "/git/commits") return {sha: headSha};
            if (path === "/git/refs") return {};
            if (path.startsWith("/pulls?")) return [];
            if (path === "/pulls") return {number: 999, html_url: "https://github.com/Kamara229/JessicaCore/pull/999"};
            if (path === "/pulls/999") return {head: {sha: headSha, repo: {full_name: "Kamara229/JessicaCore"}}, base: {ref: "main"}, state: "open"};
            if (path.endsWith(".yml")) return {id: 1};
            if (path.includes("/runs?")) return {workflow_runs: [{id: 1, event: "pull_request", head_sha: headSha,
                head_repository: {full_name: "Kamara229/JessicaCore"}, pull_requests: [{number: 999}], status: "completed", conclusion: "success"}]};
            if (path === "/git/refs/heads/main") {
                assert.equal(args.method, "PATCH");
                assert.deepEqual(args.body, {sha: headSha, force: false});
                applied++; return {object: {sha: headSha}};
            }
            throw Error("Unexpected API path " + path);
        };
        await submitPhoneChat(prepared.id, {nonce: prepared.id, baseSha: prepared.baseSha, status: "change",
            files: [{...source, content: "export const x=2;"}, {path: phoneTestPath(prepared.id), sha: null, content: "test"}]}, api);
        const result = await readPhoneChat(prepared.id, api);
        assert.equal(result.state, autoApply ? "merged" : "verified");
        assert.equal(applied, autoApply ? 1 : 0);
    }
});

test("cancellation during candidate construction prevents branch publication", async () => {
    const prepared = await preparePhoneChat("context", false, prepareApi);
    let refs = 0;
    const api = async path => {
        if (path === "/git/ref/heads/main") return {object: {sha: session.baseSha}};
        if (path === "/git/trees") { stopPhoneChat(prepared.id); return {sha: "tree"}; }
        refs++; throw Error("Should not write after cancellation");
    };
    await assert.rejects(submitPhoneChat(prepared.id, {nonce: prepared.id, baseSha: prepared.baseSha, status: "change",
        files: [{...source, content: "export const x=2;"}, {path: phoneTestPath(prepared.id), sha: null, content: "test"}]}, api), /cancelled/);
    assert.equal(refs, 0);
});
