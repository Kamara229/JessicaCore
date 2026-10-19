import test from "node:test";
import assert from "node:assert/strict";
import {partitionComparisonClaims} from "../core/validator/claim/claimUncertaintyScope.js";
import {unsupportedComparisonDuty} from "../core/validator/claim/comparisonDutyEvidence.js";

const urls=[
  "https://www.apache.org/licenses/LICENSE-2.0.html",
  "https://opensource.org/license/bsd-3-clause"
];
const answer=[
  "| Критерий | Apache 2.0 | BSD 3-Clause |",
  "| --- | --- | --- |",
  "| Патентные условия | Прямое предоставление прав | Не удалось подтвердить по прочитанному тексту |",
  "| Уведомления | Предусмотрены | Сохранение уведомлений |"
].join("\n");

test("unverified source cell does not demand a nonexistent citation",()=>{
 const input=[
  {claim:"В предоставленных данных BSD 3-Clause патентный грант не подтвержден",sourceUrl:urls[1],supported:false},
  {claim:"BSD 3-Clause разрешает распространение",sourceUrl:urls[1],supported:true}
 ];
 const clean=partitionComparisonClaims(input,answer,urls);
 assert.equal(clean.ignoredUncertainty,1);
 assert.equal(clean.claims.length,1);
});

test("categorical negatives remain rejected, never hidden",()=>{
 const items=[
  {claim:"В BSD 3-Clause отсутствуют патентные условия",sourceUrl:urls[1],supported:false},
  {claim:"В Apache патентный грант не подтвержден",sourceUrl:urls[0],supported:false}
 ];
 const clean=partitionComparisonClaims(items,answer,urls);
 assert.equal(clean.ignoredUncertainty,0);
 assert.equal(clean.claims.length,2);
});

test("lack of a mapped uncertainty cell cannot suppress unsupported facts",()=>{
 const x=partitionComparisonClaims(
  [{claim:"Патентный грант BSD не подтвержден",sourceUrl:urls[1],supported:false}],
  "| Критерий | Первый | Второй |\n| --- | --- | --- |\n| Патенты | Да | Не удалось подтвердить |",
  urls
 );
 assert.equal(x.ignoredUncertainty,0);
});

test("a copyright template cannot prove an obligation",()=>{
 assert.equal(unsupportedComparisonDuty(
  "Apache требует предоставления лицензии и NOTICE",
  "Copyright [yyyy] Licensed under the Apache License"),true);
 assert.equal(unsupportedComparisonDuty(
  "BSD требует сохранять copyright notice",
  "Redistributions of source code must retain the above copyright notice"),false);
 assert.equal(unsupportedComparisonDuty(
  "Apache содержит патентную лицензию",
  "Grant of Patent License"),false);
});
