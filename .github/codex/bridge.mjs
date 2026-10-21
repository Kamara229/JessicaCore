import {readFileSync,writeFileSync} from "node:fs";

const phase=String(process.argv[2]||"");
if(!["proposal","diff"].includes(phase))throw Error("Invalid reviewer phase");
const issue=Number(process.env.CODEX_ISSUE);
const goal=String(process.env.CODEX_GOAL||"");
const repo=process.env.GITHUB_REPOSITORY;
const code=readFileSync(phase==="proposal"?"codex-plan.txt":"codex-changes.txt","utf8").slice(0,14500).trim();
if(!code)throw Error("Codex did not produce a reviewable answer");
const githubURL="https://api.github.com/repos/"+repo+"/issues/"+issue+"/comments";
async function comment(author,message) {
    const res=await fetch(githubURL,{
        method:"POST",
        headers:{Authorization:"Bearer "+process.env.GH_TOKEN,
            Accept:"application/vnd.github+json","Content-Type":"application/json"},
        body:JSON.stringify({body:"["+author+"]\n"+message.slice(0,3000)}),
        signal:AbortSignal.timeout(15000)
    });
    if(!res.ok)throw Error("GitHub comment failed "+res.status);
}
await comment("Codex",phase==="proposal"?"Plan:\n"+code:"Patch summary:\n"+code.slice(0,2700));
const response=await fetch("https://jessicacore.onrender.com/api/codex/peer-review",{
    method:"POST",
    headers:{"X-Jessica-Token":process.env.JESSICA_APP_TOKEN,
        "Content-Type":"application/json"},
    body:JSON.stringify({goal,phase,content:code}),
    signal:AbortSignal.timeout(120000)
});
if(!response.ok)throw Error("Jessica reviewer HTTP "+response.status);
const json=await response.json();
if(json.success!==true||!json.text)throw Error("Jessica reviewer returned no verified reply");
const review=String(json.text).slice(0,3200);
writeFileSync(phase==="proposal"?"jessica-review.txt":"jessica-final-review.txt",review);
await comment("Jessica",review);
process.stdout.write("Jessica reviewed Codex "+phase+" and reported to issue\n");
