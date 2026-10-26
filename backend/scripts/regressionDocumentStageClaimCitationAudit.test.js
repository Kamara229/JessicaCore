import test from "node:test";
import assert from "node:assert/strict";
import {auditStageClaimFragmentLinks}
 from "../core/files/documentStageClaimCitationAudit.js";
import {createLosslessWindowEvidence}
 from "../core/files/documentWindowEvidence.js";

const source="Судебная практика требует проверки доказательств. "+
 "А".repeat(900)+"Окончание судебного анализа. "+
 "Б".repeat(900);
const quote="Судебная практика требует проверки доказательств.";
const fragment2=createLosslessWindowEvidence({name:"doc",text:source}).fragments[1].text.slice(0,22);
const body=(ref,value)=>
 "Выводы\n\n• Позиция автора: Правовая позиция подтверждается "+
 "[О17:Ф1:Ф"+ref+"]\n  Основание: «"+value+
 "»\n\nГраницы проверки: не доказана достоверность вывода.";

test("source-matching citation permits a quote within its actual fragment",()=>{
 const result=auditStageClaimFragmentLinks(body(1,quote),source,17);
 assert.equal(result.claims,1);
 assert.equal(result.refs,1);
 assert.equal(result.quotes,1);
 assert.equal(result.fragmentCitationsChecked,true);
 assert.equal(result.reportAuthenticityVerified,false);
});
test("wrong fragment, nonexistent ref, false window and invented quote are rejected",()=>{
 assert.equal(auditStageClaimFragmentLinks(body(2,quote),source,17),null);
 assert.equal(auditStageClaimFragmentLinks(body(999,quote),source,17),null);
 assert.equal(auditStageClaimFragmentLinks(body(1,quote),source,18),null);
 assert.equal(auditStageClaimFragmentLinks(body(1,"Выдуманное доказательство спора"),source,17),null);
});
test("multiple cited fragments each require corresponding quotation",()=>{
 const valid="• Позиция автора: Обоснование [О17:Ф1:Ф1] [О17:Ф1:Ф2]\n"+
 "  Основание: «"+quote+"»; «"+fragment2+"»";
 const checked=auditStageClaimFragmentLinks(valid,source,17);
 assert.equal(checked.claims,1);
 assert.equal(checked.refs,2);
 assert.equal(checked.quotes,2);
 assert.equal(auditStageClaimFragmentLinks(
   valid.replace("«"+fragment2+"»","«"+quote+"»"),source,17),null);
});
test("empty or invalid source and window number cannot verify citations",()=>{
 assert.equal(auditStageClaimFragmentLinks("",source,17),null);
 assert.equal(auditStageClaimFragmentLinks(body(1,quote),"",17),null);
 assert.equal(auditStageClaimFragmentLinks(body(1,quote),source,257),null);
});
