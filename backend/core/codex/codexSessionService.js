import {timingSafeEqual} from "node:crypto";

const REPO="Kamara229/JessicaCore";
const PREFIX="[Jessica Codex]";
const ALLOWED_GOALS=Object.freeze({
    language:"Improve Russian/English response-language reliability without altering code blocks or factual citations.",
    context:"Improve accurate recognition of conversation context and follow-up requests.",
    foundations:"Improve deterministic safe foundations for source-code and structured-data understanding."
});
const API="https://api.github.com";
const REQUEST_TIMEOUT_MS=15_000;
let startInFlight=false;
let latestStartAt=0;

export function codexGoals() {
    return Object.keys(ALLOWED_GOALS);
}

export function codexConfiguration() {
    return {
        ready:Boolean(process.env.JESSICA_CODEX_OWNER_KEY &&
            process.env.JESSICA_GITHUB_AUTOMATION_TOKEN),
        goals:codexGoals(),
        githubRepo:REPO
    };
}

/** The Android app token is shared with every APK; it is NOT owner authentication. */
export function validCodexOwnerKey(candidate) {
    const expected=process.env.JESSICA_CODEX_OWNER_KEY||"";
    if(typeof candidate!=="string" || candidate.length<16 || expected.length<24 ||
        candidate.length>256) return false;
    const actualBytes=Buffer.from(candidate,"utf8");
    const expectedBytes=Buffer.from(expected,"utf8");
    if(actualBytes.length!==expectedBytes.length)return false;
    return timingSafeEqual(actualBytes,expectedBytes);
}

export function validateCodexIssue(value) {
    const n=Number(value);
    return Number.isSafeInteger(n) && n>0 && n<1_000_000_000 ? n : null;
}

async function github(path,{method="GET",body}={}) {
    const token=process.env.JESSICA_GITHUB_AUTOMATION_TOKEN;
    if(!token) throw new Error("codex-github-credentials-missing");
    const response=await fetch(API+path,{
        method,
        headers:{
            Authorization:"Bearer "+token,
            Accept:"application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28",
            ...(body===undefined?{}:{"Content-Type":"application/json"})
        },
        ...(body===undefined?{}:{body:JSON.stringify(body)}),
        signal:AbortSignal.timeout(REQUEST_TIMEOUT_MS)
    });
    if(!response.ok){
        // Never log token, response body or user input.
        throw new Error("github-http-"+response.status);
    }
    if(response.status===204)return null;
    return response.json();
}
const root="/repos/"+REPO;

export async function startCodexSession(goal, api=github) {
    if(!Object.hasOwn(ALLOWED_GOALS,goal)) return {ok:false,status:400,reason:"invalid-goal"};
    if(!codexConfiguration().ready) return {ok:false,status:503,reason:"codex-not-configured"};
    if(startInFlight||Date.now()-latestStartAt<120_000)
        return {ok:false,status:429,reason:"codex-start-cooldown"};
    startInFlight=true;
    try{
        // Only one open autonomous session. No client-defined prompt, repo or workflow.
        const issues=await api(root+"/issues?state=open&per_page=60");
        const existing=issues.find(item=>!item.pull_request&&
            item.title?.startsWith(PREFIX));
        if(existing)return {ok:false,status:409,reason:"codex-session-already-running",issue:existing.number};
        const issue=await api(root+"/issues",{
            method:"POST",body:{
                title:PREFIX+" "+goal,
                body:[
                    "Autonomous source-only Codex ↔ Jessica engineering session.",
                    "Objective: "+ALLOWED_GOALS[goal],
                    "Scope: public repository source only; no private chats, attachments or secrets.",
                    "A bounded workflow opens a PR; it does not merge or deploy code."
                ].join("\n\n")
            }
        });
        try{
            await api(root+"/actions/workflows/jessica-codex.yml/dispatches",{
                method:"POST",body:{ref:"main",inputs:{issue:String(issue.number),goal}}
            });
        }catch(error){
            await api(root+"/issues/"+issue.number,{
                method:"PATCH",body:{state:"closed",state_reason:"not_planned"}
            }).catch(()=>{});
            throw error;
        }
        latestStartAt=Date.now();
        return {ok:true,status:202,issue:issue.number,url:issue.html_url,goal};
    }catch(error){
        console.warn("Codex session launch failed",{code:error.message});
        return {ok:false,status:502,reason:"codex-dispatch-failed"};
    }finally{
        startInFlight=false;
    }
}

export async function readCodexSession(issueNumber,api=github) {
    const id=validateCodexIssue(issueNumber);
    if(!id)return {ok:false,status:400,reason:"invalid-issue"};
    try{
        const issue=await api(root+"/issues/"+id);
        if(!issue.title?.startsWith(PREFIX)||issue.pull_request)
            return {ok:false,status:404,reason:"unknown-session"};
        const comments=await api(root+"/issues/"+id+"/comments?per_page=100");
        const conversation=comments.slice(-45).map(item=>{
            const raw=String(item.body||"").slice(0,3500);
            // Public issues can receive external comments. Never let an external
            // author masquerade as Codex or Jessica by prefixing a message.
            const trusted=["github-actions[bot]","Kamara229"].includes(item.user?.login);
            const role=!trusted?"External":raw.startsWith("[Codex]")?"Codex":
                raw.startsWith("[Jessica]")?"Jessica":"System";
            return {role,text:raw.replace(/^\[(?:Codex|Jessica|System)\]\s*/,""),at:item.created_at};
        });
        return {ok:true,status:200,issue:id,state:issue.state,url:issue.html_url,
            conversation};
    }catch(error){
        console.warn("Codex session read failed",{code:error.message});
        return {ok:false,status:502,reason:"codex-status-unavailable"};
    }
}

export async function stopCodexSession(issueNumber,api=github) {
    const id=validateCodexIssue(issueNumber);
    if(!id)return {ok:false,status:400,reason:"invalid-issue"};
    try{
        const issue=await api(root+"/issues/"+id);
        if(!issue.title?.startsWith(PREFIX)||issue.pull_request)
            return {ok:false,status:404,reason:"unknown-session"};
        if(issue.state==="closed")return {ok:true,status:200,issue:id};
        await api(root+"/issues/"+id,{
            method:"PATCH",body:{state:"closed",state_reason:"not_planned"}
        });
        // Best effort: cancel the active GitHub Actions runner, not just PR publication.
        try{
            const runs=await api(root+"/actions/workflows/jessica-codex.yml/runs?per_page=40");
            const active=(runs.workflow_runs||[]).filter(run=>
                (run.status==="queued"||run.status==="in_progress"||run.status==="waiting") &&
                String(run.display_title||"").startsWith("Jessica Codex #"+id+" / "));
            for(const run of active.slice(0,2)){
                await api(root+"/actions/runs/"+run.id+"/cancel",{method:"POST"}).catch(()=>{});
            }
        }catch(error){
            console.warn("Codex runner cancellation unavailable",{code:error.message});
        }
        return {ok:true,status:200,issue:id};
    }catch(error){
        console.warn("Codex session stop failed",{code:error.message});
        return {ok:false,status:502,reason:"codex-stop-unavailable"};
    }
}
