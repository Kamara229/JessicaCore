import test from "node:test";
import assert from "node:assert/strict";
import {pageDocumentText} from "../core/files/documentTextWindow.js";
import {createLosslessWindowEvidence} from "../core/files/documentWindowEvidence.js";
import {analyzeDocumentAttachments} from "../core/files/documentAnalysisFacade.js";
import {mapVerifiedClaimsToSourceSections}
 from "../core/files/documentClaimProvenance.js";

const source="Преамбула исследователя.\nВВЕДЕНИЕ\nОбоснование темы.\n"+
    "ГЛАВА 1. Правовая природа\nАнализ теории.\n"+
    "1.1 Теоретическая позиция\nНормативные предпосылки.\n"+
    "1.1.1 Судебная практика\nПозиции судов.\n"+
    "ЗАКЛЮЧЕНИЕ\nОбщие результаты.";
const pages=(text,name)=>{
    const all=[];let cursor=0;
    do{const page=pageDocumentText(text,cursor);all.push({...page,name});cursor=page.nextOffset;}
    while(cursor!==null);
    return {name,pages:all};
};
const model=async(_prompt,_task,items)=>{
    const ref=createLosslessWindowEvidence(items[0]).fragments[0];
    return JSON.stringify({sections:[{heading:"AI-generated misleading heading",findings:[{
        text:"Автор даёт характеристику исследуемого явления",
        kind:"author_claim",
        evidence:[{ref:ref.ref,quote:ref.text.slice(0,Math.min(42,ref.text.length))}]
    }]}]});
};
test("each claim maps to actual source chapter, not an invented model heading",async()=>{
    const result=await analyzeDocumentAttachments({task:"Проверь структуру",
        attachments:[pages(source,"study.docx")]},model);
    assert.equal(result.success,true);
    const provenance=result.claimProvenance;
    assert.equal(provenance.count,result.groundedClaims);
    assert.equal(provenance.sourceHeadingsHeuristic,true);
    assert.equal(provenance.logicalInferenceVerified,false);
    const nested=provenance.claims.find(x=>x.headingPath.at(-1)==="1.1.1 Судебная практика");
    assert.ok(nested);
    assert.deepEqual(nested.headingPath,[
        "ГЛАВА 1. Правовая природа",
        "1.1 Теоретическая позиция",
        "1.1.1 Судебная практика"
    ]);
    assert.ok(provenance.claims.every(x=>!/AI-generated/u.test(x.headingPath.join(" "))));
    assert.ok(provenance.claims.every(x=>/^[a-f0-9]{64}$/u.test(x.statementSha256)));
    assert.ok(provenance.claims.every(x=>!Object.hasOwn(x,"text")));
    assert.equal(result.readingCoverage.completeSourceAnalysisConfirmed,false);
});
test("invalid citations and conflicting section lineage cannot be accepted",async()=>{
    const r=await analyzeDocumentAttachments({task:"Проверь",attachments:[pages(source,"x.txt")]},model);
    assert.equal(r.success,true);
    const bad=[...r.verifiedClaims];
    bad[0]={...bad[0],refs:["О1:Ф1:Ф999"]};
    assert.equal(mapVerifiedClaimsToSourceSections(bad,r.evidenceLineage),null);
    const tampered={...r.evidenceLineage,citations:[
        {...r.evidenceLineage.citations[0],headingPath:["Подделанная глава"]},
        ...r.evidenceLineage.citations.slice(1)]};
    // The model cannot rewrite original source heading metadata.
    assert.notDeepEqual(tampered.citations[0].headingPath,
        r.evidenceLineage.citations[0].headingPath);
    assert.equal(mapVerifiedClaimsToSourceSections(r.verifiedClaims,{
        ...r.evidenceLineage,verifiedClaimCount:r.verifiedClaims.length+1}),null);
});
test("multi-file claims keep per-file identity with the same filenames",async()=>{
    const other="ВВЕДЕНИЕ\n"+"Отдельное положение. ".repeat(1300)+
        "\nГЛАВА 2. Выводы\n2.1 Частный вывод\nДругое содержание.";
    const r=await analyzeDocumentAttachments({task:"Сопоставь",
        attachments:[pages(source,"same.docx"),pages(other,"same.docx")]},model);
    assert.equal(r.success,true);
    assert.equal(r.claimProvenance.count,r.groundedClaims);
    assert.equal(new Set(r.claimProvenance.claims.map(c=>c.fileIndex)).size,2);
    for(const row of r.claimProvenance.claims){
        assert.ok(row.evidenceRefs.every(ref=>ref.startsWith(`Д${row.fileIndex}:`)));
        assert.equal(row.headingPath.at(-1),
            r.evidenceLineage.files[row.fileIndex-1].citations.find(
                x=>x.ref===row.evidenceRefs[0]).headingPath.at(-1));
    }
    assert.equal(r.claimProvenance.independentSourcesVerified,false);
});
