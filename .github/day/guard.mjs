import {execFileSync} from "node:child_process";
import {lstatSync,readFileSync} from "node:fs";
const files=execFileSync("git",["ls-files","--others","--exclude-standard","backend/scripts/"],{encoding:"utf8"}).trim().split("\n").filter(Boolean);
const permitted=/^backend\/scripts\/regressionDay(?:Language|Context|Foundations)\.test\.js$/;
if(files.length>1)throw Error("Daytime agent generated more than one file");
if(files.some(file=>!permitted.test(file)))throw Error("Daytime agent touched an unauthorized file");
for(const file of files){
 if(lstatSync(file).isSymbolicLink()||readFileSync(file,"utf8").length>4500)
  throw Error("Daytime test file violated size or file policy");
}
const modified=execFileSync("git",["diff","--name-only","HEAD"],{encoding:"utf8"}).trim();
if(modified)throw Error("Daytime automation cannot modify tracked files");
console.log("Daytime scope verified; created files:",files.length);