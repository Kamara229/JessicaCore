import test from "node:test";
import assert from "node:assert/strict";
import {parseComparisonCellBatch} from "../core/validator/claim/comparisonCellQuoteParser.js";
import {verifyComparisonCellEvidence} from "../core/validator/claim/comparisonCellEvidence.js";
import {planComparisonBatches} from "../core/validator/claim/comparisonCellBatcher.js";
import {buildComparisonCellMatrix} from "../core/validator/claim/comparisonCellMatrix.js";

const apache="https://www.apache.org/licenses/LICENSE-2.0.html";
const bsd="https://opensource.org/license/bsd-3-clause";
const source=[
  "SOURCE 1",
  "URL: "+apache,
  "SNIPPET: Each Contributor hereby grants to You a perpetual, worldwide, no-charge patent license.",
  "If You institute patent litigation, patent licenses for that Work shall terminate as of the date such litigation is filed.",
  "SOURCE 2",
  "URL: "+bsd,
  "SNIPPET: Redistributions of source code must retain the above copyright notice and disclaimer.",
  "Redistributions in binary form must reproduce the above copyright notice and disclaimer."
].join("\n");

test("both independent patent facts need different source clauses",()=>{
 const cell={id:1,criterion:"Патентные условия",
  value:"Лицензия предоставляет патентный грант и предусматривает его прекращение",
  sourceUrl:apache,source:0};
 const grant="Each Contributor hereby grants to You a perpetual, worldwide, no-charge patent license";
 const ending="patent licenses for that Work shall terminate as of the date such litigation is filed";
 assert.equal(verifyComparisonCellEvidence(cell,[grant,ending],source).supported,true);
 assert.equal(verifyComparisonCellEvidence(cell,[grant],source).supported,false);
});

test("source and binary copyright duties must be separately evidenced",()=>{
 const cell={id:2,criterion:"Уведомления и авторство",
  value:"В исходной и бинарной форме нужно сохранять уведомление об авторских правах",
  sourceUrl:bsd,source:1};
 const sourceDuty="Redistributions of source code must retain the above copyright notice and disclaimer";
 const binaryDuty="Redistributions in binary form must reproduce the above copyright notice and disclaimer";
 assert.equal(verifyComparisonCellEvidence(cell,[sourceDuty,binaryDuty],source).supported,true);
 assert.equal(verifyComparisonCellEvidence(cell,[sourceDuty],source).supported,false);
});

test("all proof IDs must exist and be attributed",()=>{
 assert.equal(parseComparisonCellBatch('{"c":[[1,["quote"]],[2,[]]]}',[1,2]).valid,true);
 assert.equal(parseComparisonCellBatch('{"c":[[1,"quote"],[1,"duplicate"]]}',[1,2]).valid,false);
 assert.equal(parseComparisonCellBatch('{"claims":[{"id":1,"evidence":"one"},{"id":2,"quotes":[]}]}',[1,2]).valid,true);
 assert.equal(parseComparisonCellBatch('{"c":[[1,["quote"]]]}',[1,2]).valid,false);
 assert.equal(parseComparisonCellBatch('{"c":[[1,["quote"]],[3,[]]]}',[1,2]).valid,false);
});

test("unknown patent cell is excluded without omitting asserted facts",()=>{
 const answer=[
  "| Критерий | Apache 2.0 | BSD 3-Clause |",
  "| --- | --- | --- |",
  "| Уведомления | Требуется сохранять копирайт | Нужно сохранять копирайт |",
  "| Патентные условия | Есть грант | Не удалось найти подтверждающую выдержку |"
 ].join("\n");
 const matrix=buildComparisonCellMatrix(answer,[apache,bsd]);
 assert.ok(matrix);
 assert.equal(matrix.checks.length,3);
});

test("token reservations stay bounded even for a wide table",()=>{
 for(const n of [1,6,9,12,18]){
  const m={checks:Array.from({length:n},(_,i)=>({id:i+1}))};
  const schedule=planComparisonBatches(m);
  assert.ok(schedule);
  assert.equal(schedule.chunks.flat().length,n);
  assert.ok(schedule.chunks.every(group=>group.length<=5));
  assert.ok(schedule.chunks.length*schedule.maxTokens<=840);
 }
});
