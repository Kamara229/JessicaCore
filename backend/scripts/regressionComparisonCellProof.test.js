import test from "node:test";
import assert from "node:assert/strict";
import {buildComparisonCellMatrix} from "../core/validator/claim/comparisonCellMatrix.js";
import {checkRequestedComparisonCoverage} from "../core/validator/claim/comparisonRequestedCoverage.js";
import {verifyComparisonCellProof} from "../core/validator/claim/comparisonCellProof.js";

const urls=["https://www.apache.org/licenses/LICENSE-2.0.html",
"https://opensource.org/license/bsd-3-clause"];
const answer=[
"| Критерий | Apache 2.0 | BSD 3-Clause |",
"| --- | --- | --- |",
"| Использование и изменения | Разрешено | Разрешено |",
"| Распространение | Разрешено | Разрешено |",
"| Уведомления и авторство | Нужно сохранить уведомления | Нужно сохранить copyright |",
"| Патентные условия | Есть грант | Не удалось подтвердить по тексту |",
"| Ограничения | Не разрешено использовать товарный знак | Нельзя рекламировать именем |"
].join("\n");
const task="Сравни использование, распространение, уведомления, патентные условия и ограничения";
test("creates exactly one check for every factual cell, not unknown",()=>{
 const m=buildComparisonCellMatrix(answer,urls);
 assert.ok(m);
 assert.equal(m.totalRows,5);
 assert.equal(m.checks.length,9);
 assert.equal(m.checks.filter(c=>c.source===1).length,4);
});
test("both requested sources and requested topics must be covered",()=>{
 const m=buildComparisonCellMatrix(answer,urls);
 assert.equal(checkRequestedComparisonCoverage(task,m).valid,true);
 const noPatents={...m,criteria:m.criteria.filter(x=>!x.includes("Патент"))};
 assert.equal(checkRequestedComparisonCoverage(task,noPatents).valid,false);
});
test("cannot guess which source is which when headers ambiguous",()=>{
 const bad=answer.replace("Apache 2.0","Первый").replace("BSD 3-Clause","Второй");
 assert.equal(buildComparisonCellMatrix(bad,urls),null);
});
test("does not silently pass missing cell IDs",()=>{
 const matrix=buildComparisonCellMatrix(answer,urls);
 const partial={c:[[1,"Some legal text"]]}; 
 assert.equal(verifyComparisonCellProof(JSON.stringify(partial),matrix,"").valid,false);
});
test("reject duplicate IDs and malformed model JSON",()=>{
 const m=buildComparisonCellMatrix(answer,urls);
 assert.equal(verifyComparisonCellProof('{"c":[[1,"a"],[1,"b"]]}',m,"").valid,false);
 assert.equal(verifyComparisonCellProof('not json',m,"").valid,false);
});
test("source reversal preserves attribution",()=>{
 const swapped=answer.replace("Apache 2.0 | BSD 3-Clause","BSD 3-Clause | Apache 2.0")
 .replace("Есть грант | Не удалось подтвердить по тексту",
 "Не удалось подтвердить по тексту | Есть грант");
 const m=buildComparisonCellMatrix(swapped,urls);
 assert.ok(m);
 assert.equal(m.checks.some(c=>c.source===0&&c.criterion==="Патентные условия"),true);
});
