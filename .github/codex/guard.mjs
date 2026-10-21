import {execFileSync} from "node:child_process";
import {lstatSync,readFileSync} from "node:fs";

const changed=execFileSync("git",["diff","--name-only","HEAD"],{encoding:"utf8"}).trim().split("\n").filter(Boolean);
const untracked=execFileSync("git",["ls-files","--others","--exclude-standard"],{encoding:"utf8"}).trim().split("\n").filter(Boolean);
const temporary=new Set(["codex-plan.txt","jessica-review.txt","codex-implementation.txt","codex-changes.txt","jessica-final-review.txt","codex-pr-body.md"]);
const files=[...new Set([...changed,...untracked])].filter(path=>!temporary.has(path));
const allowed=/^backend\/core\/(?:foundations|language|chat)\/[A-Za-z0-9_-]+\.js$|^backend\/scripts\/regression[A-Za-z0-9_-]+\.test\.js$/;
if(!files.length) {
    process.stdout.write("No code changes; nothing to publish.\n");
    process.exit(3);
}
if(files.length>8)throw Error("Exceeded 8 file limit");
let chars=0;
for(const filename of files){
    if(!allowed.test(filename))throw Error("Protected path or unsanctioned change: "+filename);
    if(lstatSync(filename).isSymbolicLink())throw Error("Symlinks forbidden: "+filename);
    const value=readFileSync(filename,"utf8");
    if(value.includes("\u0000"))throw Error("Binary source forbidden");
    chars+=value.length;
}
if(chars>50000)throw Error("Code change exceeded size limit");
process.stdout.write("Scope approved: "+files.join(", ")+"\n");
