import {createHash,createHmac,timingSafeEqual} from "node:crypto";
import {documentStageTaskSha256} from "./documentStageTaskIdentity.js";

const HEX=/^[a-f0-9]{64}$/u;
const MAX_BYTES=240_000;
const MAX_WINDOWS=16;
const hash=value=>createHash("sha256").update(value,"utf8").digest("hex");
const key=()=>/^[a-f0-9]{64}$/u.test(
    String(process.env.JESSICA_STAGE_ATTESTATION_KEY||""))?
    Buffer.from(process.env.JESSICA_STAGE_ATTESTATION_KEY,"hex"):null;

function payloadFor({source,task,name,plan,rawResponses}) {
    const sha=hash(source);
    if(typeof source!=="string"||!source.length||
       typeof name!=="string"||!name||name.length>100||
       !plan?.complete||!Array.isArray(plan.windows)||
       plan.windows.length<2||plan.windows.length>MAX_WINDOWS||
       !Array.isArray(rawResponses)||!rawResponses.length||
       rawResponses.length>=plan.windows.length||
       rawResponses.some(x=>typeof x!=="string"||
           !x.trim()||Buffer.byteLength(x,"utf8")>32_000))
        return null;
    const taskSha256=documentStageTaskSha256(task);
    if(!taskSha256||!HEX.test(sha))return null;
    const windows=plan.windows.slice(0,rawResponses.length).map(w=>({
        number:w.number,sectionIndex:w.sectionIndex,
        startOffset:w.startOffset,endOffset:w.endOffset,
        windowSha256:hash(w.text)
    }));
    const body={version:1,sourceSha256:sha,sourceChars:source.length,
        taskSha256,name,windows,rawResponses:[...rawResponses]};
    const serialized=JSON.stringify(body);
    return Buffer.byteLength(serialized,"utf8")<=MAX_BYTES?{
        body,serialized
    }:null;
}

/** Only responses that passed source/quote validation may be sealed by caller. */
export function sealDocumentPagedResume({source,task,name,plan,rawResponses}){
    const secret=key();
    if(!secret)return null;
    const built=payloadFor({source,task,name,plan,rawResponses});
    if(!built)return null;
    return {...built.body,authTag:"v1:"+
        createHmac("sha256",secret).update(built.serialized,"utf8")
            .digest("hex")};
}

/** Rebuild the expected source/task/window manifest; verify MAC fail-closed.
 * Resumed model responses MUST still be revalidated by the main analysis.
 */
export function openDocumentPagedResume({checkpoint,source,task,name,plan}){
    const secret=key();
    if(!secret||!checkpoint||typeof checkpoint!=="object"||
       Array.isArray(checkpoint)||typeof checkpoint.authTag!=="string"||
       !/^v1:[a-f0-9]{64}$/u.test(checkpoint.authTag)||
       Object.keys(checkpoint).length!==8||
       !Array.isArray(checkpoint.windows)||!Array.isArray(checkpoint.rawResponses))
        return null;
    const built=payloadFor({source,task,name,plan,
        rawResponses:checkpoint.rawResponses});
    if(!built)return null;
    const expected=createHmac("sha256",secret)
        .update(built.serialized,"utf8").digest("hex");
    if(!timingSafeEqual(Buffer.from(expected,"hex"),
       Buffer.from(checkpoint.authTag.slice(3),"hex")))return null;
    if(checkpoint.version!==built.body.version||
       checkpoint.sourceSha256!==built.body.sourceSha256||
       checkpoint.sourceChars!==built.body.sourceChars||
       checkpoint.taskSha256!==built.body.taskSha256||
       checkpoint.name!==built.body.name||
       checkpoint.windows.length!==built.body.windows.length||
       checkpoint.windows.some((w,i)=>
            !w||typeof w!=="object"||Array.isArray(w)||
            Object.keys(w).length!==5||
            Object.keys(built.body.windows[i]).some(k=>
                w[k]!==built.body.windows[i][k])))return null;
    return {rawResponses:[...built.body.rawResponses],
        verifiedPrefix:built.body.windows.length,
        sourceSha256:built.body.sourceSha256};
}
