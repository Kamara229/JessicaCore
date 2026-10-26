import {pageDocumentText} from "../../core/files/documentTextWindow.js";
import {createLosslessWindowEvidence}
 from "../../core/files/documentWindowEvidence.js";
import {materializeVerifiedDocumentOverflowBatches}
 from "../../core/files/documentOverflowBatchBulkMaterialization.js";
import {makeDocumentStressSource,hashDocumentStressSource}
 from "./documentStageClaimStressCorpus.js";

export const httpSource=makeDocumentStressSource();
export const httpSourceSha256=hashDocumentStressSource(httpSource);
export const httpTask="Исследуй 256 окон и сохрани дословные цитаты";
const prepared=materializeVerifiedDocumentOverflowBatches(
    httpSource,httpSourceSha256);
if(!prepared?.verified?.complete||
    prepared.verified.totalWindows!==256||
    prepared.batches.length!==16)
    throw Error("HTTP test must use complete 256-window source");
const pages=[];
for(let offset=0;offset!==null;){
    const page=pageDocumentText(httpSource,offset);
    pages.push({...page,name:"huge.docx"});
    offset=page.nextOffset;
}
export const httpAttachment={name:"huge.docx",pages,
    overflowBatches:prepared.batches};

/** Pure provider simulation. A claim is useful only when quoted and audited. */
export async function syntheticDocumentAnswer(_prompt,_task,attachments){
    const first=createLosslessWindowEvidence(attachments[0])?.fragments[0];
    const quote=first?.text.split(/\r?\n/u)
        .find(line=>line.trim().length>=6)?.trim().slice(0,32);
    if(!quote||!first?.ref)throw Error("Missing source quote");
    return JSON.stringify({sections:[{heading:"Основания",findings:[{
        kind:"author_claim",
        text:"Источник подтверждает процитированный фрагмент",
        evidence:[{ref:first.ref,quote}]
    }]}]});
}
