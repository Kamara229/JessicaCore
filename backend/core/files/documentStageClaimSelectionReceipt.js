import {createHash,createHmac,timingSafeEqual} from "node:crypto";
import {documentStageTaskSha256} from "./documentStageTaskIdentity.js";

const HASH=/^[a-f0-9]{64}$/u;
const sha=value=>createHash("sha256").update(value,"utf8").digest("hex");

function serialize(page,positions,task){
    const taskSha256=documentStageTaskSha256(task);
    if(!taskSha256||!HASH.test(String(page?.sourceSha256||""))||
       !Array.isArray(page?.claims)||!page.claims.length||
       !Number.isSafeInteger(page.claimOffset)||page.claimOffset<0||
       !Number.isSafeInteger(page.totalClaims)||page.totalClaims<1||
       page.returnedClaims!==page.claims.length||
       (!Number.isSafeInteger(page.nextClaimOffset)&&page.nextClaimOffset!==null)||
       !Array.isArray(positions)||!positions.length||positions.length>6)return null;
    let prior=0;
    for(const position of positions){
        if(!Number.isSafeInteger(position)||position<=prior||
           position>page.claims.length)return null;
        prior=position;
    }
    const pageSha256=sha(JSON.stringify(page.claims));
    const payload=JSON.stringify(["Jessica.StageClaimSelection.v1",
        page.sourceSha256,taskSha256,page.claimOffset,page.nextClaimOffset,
        page.totalClaims,page.returnedClaims,pageSha256,positions]);
    if(Buffer.byteLength(payload,"utf8")>4096)return null;
    return {payload,taskSha256,pageSha256};
}

/** Authenticate ranking positions and page identity, not the truth of claims. */
export function signStageClaimSelection(page,positions,task,
    secret=process.env.JESSICA_STAGE_ATTESTATION_KEY){
    if(typeof secret!=="string"||!HASH.test(secret))return null;
    const encoded=serialize(page,positions,task);
    if(!encoded)return null;
    return {version:1,sourceSha256:page.sourceSha256,
        taskSha256:encoded.taskSha256,pageSha256:encoded.pageSha256,
        claimOffset:page.claimOffset,nextClaimOffset:page.nextClaimOffset,
        totalClaims:page.totalClaims,pageClaims:page.returnedClaims,
        selectedPositions:[...positions],
        authTag:"v1:"+createHmac("sha256",Buffer.from(secret,"hex"))
            .update(encoded.payload,"utf8").digest("hex")};
}

export function verifyStageClaimSelection(page,receipt,task,
    secret=process.env.JESSICA_STAGE_ATTESTATION_KEY){
    if(!receipt||receipt.version!==1||
       typeof receipt.authTag!=="string"||
       !/^v1:[a-f0-9]{64}$/u.test(receipt.authTag))return false;
    const expected=signStageClaimSelection(
        page,receipt.selectedPositions,task,secret);
    if(!expected)return false;
    for(const key of ["version","sourceSha256","taskSha256",
        "pageSha256","claimOffset","nextClaimOffset",
        "totalClaims","pageClaims"]){
        if(receipt[key]!==expected[key])return false;
    }
    return timingSafeEqual(Buffer.from(receipt.authTag,"utf8"),
        Buffer.from(expected.authTag,"utf8"));
}
