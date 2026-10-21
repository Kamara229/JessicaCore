import {readFileSync} from "node:fs";

const issue=Number(process.env.CODEX_ISSUE);
const goal=String(process.env.CODEX_GOAL||"");
const owner=process.env.GITHUB_REPOSITORY?.split("/")[0];
const validGoals=new Set(["language","context","foundations"]);
if(!Number.isSafeInteger(issue)||issue<=0||!validGoals.has(goal)||
   process.env.GITHUB_REPOSITORY!=="Kamara229/JessicaCore"){
    process.stderr.write("Invalid automation objective or target.\n");
    process.exit(1);
}
const response=await fetch("https://api.github.com/repos/"+process.env.GITHUB_REPOSITORY+"/issues/"+issue,{
    headers:{Authorization:"Bearer "+process.env.GH_TOKEN,Accept:"application/vnd.github+json"},
    signal:AbortSignal.timeout(15000)
});
if(!response.ok)process.exit(1);
const data=await response.json();
if(data.pull_request||data.state!=="open"||data.title!=="[Jessica Codex] "+goal||
   data.user?.login!==owner) {
    process.stderr.write("Issue is not a trusted, open Codex session.\n");
    process.exit(1);
}
process.stdout.write("Authorized bounded Codex session #"+issue+" / "+goal+"\n");
