import {execFileSync} from "node:child_process";
import {existsSync} from "node:fs";
const scripts=[
"regressionLanguageService.test.js","regressionLocaleGeneration.test.js",
"regressionProgrammingFoundation.test.js","regressionProgrammingAnswerSafety.test.js",
"regressionContextDataFoundations.test.js","regressionLocalConversation.test.js",
"regressionLearningSafety.test.js"
].map(f=>"backend/scripts/"+f).filter(existsSync);
if(!scripts.length)throw Error("Missing pinned regression tests");
execFileSync(process.execPath,["--test",...scripts],{stdio:"inherit",timeout:180000});
