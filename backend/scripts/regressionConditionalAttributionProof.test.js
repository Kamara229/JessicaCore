import test from "node:test";
import assert from "node:assert/strict";
import {buildComparisonClauseIndex} from "../core/validator/claim/comparisonClauseIndex.js";
import {rejectsUnsupportedLegalEffect} from "../core/validator/claim/comparisonLegalEffectGate.js";

const source=("Section conditions. ").repeat(55)+
 "You must retain all copyright and attribution notices from the source form. "+
 "If the Work includes a NOTICE file, distributed derivatives must include a readable copy of the attribution notices contained within the file.";
const run={results:[
 {tool:"web_fetch",success:true,data:{url:"https://a.example/source",content:source}},
 {tool:"web_fetch",success:true,data:{url:"https://b.example/source",content:
  "Redistributions of source code must retain the above copyright notice. "+
  "Redistributions in binary form must reproduce the notice."}}
]};
test("source index provides independent operative notice clauses",()=>{
 const indexed=buildComparisonClauseIndex(run);
 assert.ok(indexed.some(x=>x.source===0&&x.criterion==="notices"));
 assert.ok(indexed.some(x=>x.source===1&&x.criterion==="notices"));
});
test("a readable copy is not automatically a requirement to retain the container file",()=>{
 const quotes=["You must retain all copyright and attribution notices from the source form",
 "distributed derivatives must include a readable copy of the attribution notices"];
 assert.equal(rejectsUnsupportedLegalEffect({
  criterion:"Уведомления и авторство",
  value:"Необходимо сохранять файл NOTICE и включать его текст"
 },quotes),true);
 assert.equal(rejectsUnsupportedLegalEffect({
  criterion:"Уведомления и авторство",
  value:"Необходимо сохранять уведомления об авторских правах"
 },quotes),false);
});
