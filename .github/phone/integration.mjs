// Exercise the actual gate against controlled fixture commits, without publishing or merging anything.
import {execFileSync, spawnSync} from "node:child_process";
import {mkdtempSync, mkdirSync, writeFileSync, copyFileSync, rmSync} from "node:fs";
import {join, dirname, resolve} from "node:path";
import {tmpdir} from "node:os";
import assert from "node:assert/strict";
import {PHONE_GOALS, phoneTestPath} from "../../backend/core/codex/phoneChatPolicy.js";

const fixture = mkdtempSync(join(tmpdir(), "phone-gate-fixture-"));
const write = (path, text) => { const dest = join(fixture, path); mkdirSync(dirname(dest), {recursive: true}); writeFileSync(dest, text); };
const git = (...args) => execFileSync("git", args, {cwd: fixture, encoding: "utf8"}).trim();
const id = "12345678-abcd-4abc-8abc-123456789012";
try {
    git("init", "--quiet"); git("config", "user.name", "Fixture"); git("config", "user.email", "fixture@example.invalid");
    write("backend/package.json", '{"type":"module"}');
    for (const path of Object.values(PHONE_GOALS).flat()) write(path, "export const value = 1;\n");
    for (const path of [".github/phone/verify.mjs", "backend/core/codex/phoneChatPolicy.js"]) {
        mkdirSync(dirname(join(fixture, path)), {recursive: true});
        copyFileSync(resolve(path), join(fixture, path));
    }
    const pinned = ["regressionLanguageService.test.js", "regressionLocaleGeneration.test.js",
        "regressionProgrammingFoundation.test.js", "regressionProgrammingAnswerSafety.test.js",
        "regressionContextDataFoundations.test.js", "regressionLocalConversation.test.js", "regressionLearningSafety.test.js"];
    for (const name of pinned) write("backend/scripts/" + name, `
import test from 'node:test'; import assert from 'node:assert/strict';
import {writeFileSync,existsSync} from 'node:fs'; import {networkInterfaces} from 'node:os';
test('runner has no host credential, writable source, external interface or .env',()=>{
 assert.equal(process.env.PHONE_HOST_SECRET,undefined);
 assert.equal(process.getuid(),1000);
 assert.ok(Object.keys(networkInterfaces()).every(name=>name==='lo'));
 assert.equal(existsSync('backend/.env'),false);
 assert.throws(()=>writeFileSync('backend/should-not-write','x'));
});\n`);
    // Deliberately present on the host; the gate must exclude it from both containers.
    write("backend/.env", "PHONE_HOST_SECRET=fixture-only\n");
    git("add", "."); git("commit", "-qm", "fixture base");
    const base = git("rev-parse", "HEAD");
    write(PHONE_GOALS.context[0], "export const value = 2;\n");
    write(phoneTestPath(id), "import test from 'node:test'; import assert from 'node:assert/strict';\nimport {value} from '../core/chat/conversationContext.js';\ntest('demonstrates the fixed behavior',()=>assert.equal(value,2));\n");
    git("add", "."); git("commit", "-qm", "fixture candidate");
    const head = git("rev-parse", "HEAD");
    // The executable must still run from the original trusted checkout.
    git("checkout", "--quiet", base);
    const result = spawnSync(process.execPath, [".github/phone/verify.mjs"], {cwd: fixture,
        env: {...process.env, PHONE_HEAD: head, PHONE_BASE: base, PHONE_BRANCH: `phone-chat/${id}`, PHONE_HOST_SECRET: "fixture-only"},
        encoding: "utf8", timeout: 240000, maxBuffer: 2_000_000});
    assert.equal(result.status, 0, result.stdout + result.stderr);
    assert.match(result.stdout, /Verified: new test fails on base/);
    console.log("PASS: real Docker gate proves baseline failure, candidate success, read-only source, no network, unprivileged uid and no host secrets.");
} finally { rmSync(fixture, {recursive: true, force: true}); }
