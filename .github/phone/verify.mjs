// Executed from the trusted PR BASE checkout, never from the model's candidate.
import {execFileSync, spawnSync} from "node:child_process";
import {mkdtempSync, mkdirSync, writeFileSync, cpSync, rmSync} from "node:fs";
import {tmpdir} from "node:os";
import {join, dirname} from "node:path";
import {PHONE_GOALS, phoneTestPath, validatePhonePatch} from "../../backend/core/codex/phoneChatPolicy.js";

const {PHONE_HEAD: head, PHONE_BASE: base, PHONE_BRANCH: branch} = process.env;
if (![head, base].every(value => /^[a-f0-9]{40}$/.test(value || ""))) throw Error("Invalid commit");
const id = /^phone-chat\/([a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12})$/.exec(branch || "")?.[1];
if (!id) throw Error("Invalid candidate branch");
const git = args => execFileSync("git", args, {encoding: "utf8", maxBuffer: 2_000_000});
if (git(["rev-parse", `${head}^`]).trim() !== base) throw Error("Candidate must be a single change on this base");
const paths = git(["diff", "--name-only", "--no-renames", base, head]).trim().split("\n").filter(Boolean);
const sources = [...new Set(Object.values(PHONE_GOALS).flat())].map(path => ({path,
    sha: git(["rev-parse", `${base}:${path}`]).trim(), content: git(["show", `${base}:${path}`])}));
const files = paths.map(path => {
    const mode = git(["ls-tree", head, "--", path]).split(" ")[0];
    if (mode !== "100644") throw Error("Only ordinary source files may change");
    return {path, sha: sources.find(source => source.path === path)?.sha || null,
        content: git(["show", `${head}:${path}`])};
});
validatePhonePatch({nonce: id, baseSha: base, status: "change", files}, {id, baseSha: base, sources});
const testFile = phoneTestPath(id);
const fixedTests = ["regressionLanguageService.test.js", "regressionLocaleGeneration.test.js",
    "regressionProgrammingFoundation.test.js", "regressionProgrammingAnswerSafety.test.js",
    "regressionContextDataFoundations.test.js", "regressionLocalConversation.test.js", "regressionLearningSafety.test.js"]
    .map(name => "backend/scripts/" + name);
const work = mkdtempSync(join(tmpdir(), "jessica-phone-"));
try {
    for (const version of ["base", "fixed"]) {
        const directory = join(work, version);
        mkdirSync(directory);
        // Only the public backend is mounted; no .git, credentials, app keys or runner home.
        cpSync("backend", join(directory, "backend"), {recursive: true,
            filter: source => !/(?:^|\/)\.env(?:\.|$)/.test(source)});
        for (const file of files) {
            if (version === "base" && file.path !== testFile) continue;
            const target = join(directory, file.path);
            mkdirSync(dirname(target), {recursive: true});
            writeFileSync(target, file.content);
        }
    }
    const run = (version, command) => spawnSync("docker", ["run", "--rm", "--network", "none",
        "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=128",
        "--memory=768m", "--cpus=1", "--user=1000:1000", "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
        "-v", `${join(work, version)}:/workspace:ro`, "-w", "/workspace", "node:22-bookworm-slim",
        "node", ...command], {encoding: "utf8", timeout: 180000, maxBuffer: 2_000_000});
    for (const file of files) {
        const syntax = run("fixed", ["--check", file.path]);
        if (syntax.error || syntax.status !== 0) throw Error("Candidate syntax check failed");
    }
    const baseline = run("base", ["--test", testFile]);
    if (baseline.error || baseline.signal || baseline.status !== 1 ||
        !/ERR_ASSERTION|AssertionError/.test(baseline.stdout + baseline.stderr))
        throw Error("New test must demonstrate an assertion failure on the original code");
    const corrected = run("fixed", ["--test", ...fixedTests, testFile]);
    process.stdout.write((corrected.stdout + corrected.stderr).slice(-40000));
    if (corrected.error || corrected.status !== 0) throw Error("Candidate failed independent regressions");
    console.log("Verified: new test fails on base, all pinned tests pass on candidate. No Learning success was fabricated.");
} finally { rmSync(work, {recursive: true, force: true}); }
