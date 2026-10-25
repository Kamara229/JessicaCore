import test from "node:test";
import assert from "node:assert/strict";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";
import {buildDocumentEvidenceLineage,scopeDocumentEvidenceLineage}
 from "../core/files/documentEvidenceLineage.js";

const source="Преамбула научного исследования.\nВВЕДЕНИЕ\nИсходные положения.\n"+
 "ГЛАВА 1. Теоретические основы\nАвтор анализирует проблему.\n"+
 "1.1 Предмет правового регулирования\nПравовые доводы.\n"+
 "1.1.1 Историческое развитие\nРанее действовало иное правило.\n"+
 "ЗАКЛЮЧЕНИЕ\nОбщий вывод.";
const pages=(text,name)=>{
    const batches=[];let at=0;
    do{const page=pageDocumentText(text,at);batches.push({...page,name});at=page.nextOffset;}
    while(at!==null);
    return {name,pages:batches};
};
const model=async(_prompt,_task,docs)=>{
    const source=createLosslessWindowEvidence(docs[0]).fragments[0];
    return JSON.stringify({sections:[{heading:"Содержание",findings:[{
        kind:"author_claim",text:"Автор приводит проверяемую позицию",
        evidence:[{ref:source.ref,quote:source.text.slice(0,Math.min(50,source.text.length))}]
    }]}]});
};
test("citations keep heading ancestry and exact absolute source offsets",async()=>{
    const r=await analyzeDocumentAttachments({task:"Проанализируй",
        attachments:[pages(source,"thesis.docx")]},model);
    assert.equal(r.success,true);
    const lineage=r.evidenceLineage;
    assert.ok(lineage.citationCount>=5);
    assert.equal(lineage.verifiedClaimCount,r.groundedClaims);
    assert.equal(lineage.sourceSha256,r.readingCoverage.sourceSha256);
    assert.equal(lineage.headingHierarchyHeuristic,true);
    assert.equal(lineage.semanticCorrectnessConfirmed,false);
    const nested=lineage.citations.find(x=>x.headingPath.some(t=>t.startsWith("1.1.1")));
    assert.ok(nested);
    assert.deepEqual(nested.headingPath,[
        "ГЛАВА 1. Теоретические основы",
        "1.1 Предмет правового регулирования",
        "1.1.1 Историческое развитие"
    ]);
    assert.ok(source.slice(nested.startOffset,nested.endOffset).length>0);
    assert.ok(lineage.citations.every(c=>c.verifiedClaimMentions>=1));
});

test("false section placement, forged refs and unverified quotes fail closed",async()=>{
    const r=await analyzeDocumentAttachments({task:"Прочитай",
        attachments:[pages(source,"thesis.docx")]},model);
    assert.equal(r.success,true);
    const args=[r.sectionCoverage,r.readingCoverage,r.citationSpans,
        r.verifiedClaims,r.quoteRanges];
    assert.ok(buildDocumentEvidenceLineage(...args));
    const forged=r.citationSpans.map((s,i)=>i===0?{...s,sectionIndex:99}:s);
    assert.equal(buildDocumentEvidenceLineage(r.sectionCoverage,r.readingCoverage,
        forged,r.verifiedClaims,r.quoteRanges),null);
    const bogus=r.verifiedClaims.map((s,i)=>i===0?{...s,refs:["О1:Ф1:Ф999"]}:s);
    assert.equal(buildDocumentEvidenceLineage(r.sectionCoverage,r.readingCoverage,
        r.citationSpans,bogus,r.quoteRanges),null);
    assert.equal(buildDocumentEvidenceLineage(r.sectionCoverage,r.readingCoverage,
        r.citationSpans,r.verifiedClaims,[...r.quoteRanges,
            {...r.quoteRanges[0],startOffset:-1}]),null);
});

test("same-named different files never share citation identity or ancestry",async()=>{
    const a=source;
    const b="ВВЕДЕНИЕ\n"+"Второй правовой анализ. ".repeat(1100)+
      "\nГЛАВА 2. Аналитический материал\n"+
      "2.1 Обобщение практики\nВыводы.";
    const r=await analyzeDocumentAttachments({task:"Сопоставь",
       attachments:[pages(a,"same.docx"),pages(b,"same.docx")]},model);
    assert.equal(r.success,true);
    assert.equal(r.evidenceLineage.files.length,2);
    const [one,two]=r.evidenceLineage.files;
    assert.notEqual(one.sourceSha256,two.sourceSha256);
    assert.ok(one.citations.every(row=>row.ref.startsWith("Д1:")));
    assert.ok(two.citations.every(row=>row.ref.startsWith("Д2:")));
    const all=[...one.citations,...two.citations].map(row=>row.ref);
    assert.equal(new Set(all).size,all.length);
    assert.ok(one.citations.some(row=>row.headingPath.at(-1).startsWith("1.1")));
    assert.ok(two.citations.some(row=>row.headingPath.at(-1).startsWith("2.1")));
    assert.equal(r.evidenceLineage.semanticCorrectnessConfirmed,false);
    assert.equal(scopeDocumentEvidenceLineage({
       ...one,sourceSha256:one.sourceSha256},1,"0".repeat(64)),null);
});
