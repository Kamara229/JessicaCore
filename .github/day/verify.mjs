const expectedRepo="Kamara229/JessicaCore";
const goal=String(process.env.DAY_GOAL||"");
const issue=Number(process.env.DAY_ISSUE);
if (process.env.GITHUB_REPOSITORY!==expectedRepo ||
    !["auto","language","context","foundations"].includes(goal) ||
    !Number.isSafeInteger(issue) || issue<0) throw Error("Untrusted daytime session");
if(issue===0){
    // GitHub UI dispatch is authenticated by GitHub itself; no server token
    // or owner key is needed. Only the repository owner may use this path.
    if(process.env.GITHUB_ACTOR!=="Kamara229")throw Error("Owner-only direct GitHub dispatch");
    console.log("Verified direct owner launch through GitHub Actions UI");
    process.exit(0);
}
const response=await fetch("https://api.github.com/repos/"+expectedRepo+"/issues/"+issue,{
    headers:{Authorization:"Bearer "+process.env.GH_TOKEN,Accept:"application/vnd.github+json"},
    signal:AbortSignal.timeout(15000)
});
if(!response.ok)throw Error("Cannot verify issue");
const data=await response.json();
if(data.user?.login!=="Kamara229"||data.pull_request||data.state!=="open"||
    data.title!=="[Jessica Codex] day:"+goal) throw Error("Day session not authorized by owner");
console.log("Verified bounded API-free daytime session",issue,goal);