import test from "node:test";
import assert from "node:assert/strict";
import { extractNoticeDutyExcerpts } from "../core/validator/claim/noticeDutyExcerpts.js";
import { buildClaimEvidenceContext } from "../core/validator/claim/claimEvidenceContext.js";
import { buildComparisonEvidenceGrid } from "../core/composer/comparisonEvidenceGrid.js";

const apache=[
  "Apache License Version 2.0. TERMS AND CONDITIONS.",
  "4. Redistribution. You may reproduce and distribute copies.",
  "You must give any other recipients of the Work a copy of this License.",
  "You must retain, in Source form of Derivative Works that You distribute,",
  "all copyright, patent, trademark and attribution notices from the Work;",
  "If the Work includes a NOTICE text file as part of its distribution,",
  "then any derivative works that You distribute must include a readable",
  "copy of the attribution notices contained within such NOTICE file.",
  "How to apply the Apache License. Copyright [yyyy] [owner]."
].join(" ");
const bsd=[
  "Redistribution and use in source and binary forms are permitted.",
  "Redistributions of source code must retain the above copyright notice,",
  "this list of conditions and the following disclaimer.",
  "Redistributions in binary form must reproduce the above copyright notice."
].join(" ");
const task="Сравни условия сохранения уведомлений и распространения лицензий";
const sources=[
 {tool:"web_fetch",success:true,data:{url:"https://www.apache.org/licenses/LICENSE-2.0.html",title:"Apache License",content:apache,truncated:false}},
 {tool:"web_fetch",success:true,data:{url:"https://opensource.org/license/bsd-3-clause",title:"BSD",content:bsd,truncated:false}}
];

test("source-first duty excerpts retain explicit clauses and conditional notices",()=>{
  const excerpts=extractNoticeDutyExcerpts(apache,task);
  assert.ok(excerpts.length>0);
  assert.ok(excerpts.some(x=>x.includes("You must retain")));
  assert.ok(excerpts.some(x=>x.includes("If the Work includes")));
  assert.ok(excerpts.every(x=>apache.includes(x)));
});

test("copyright boilerplate cannot substitute for a source obligation",()=>{
  assert.equal(extractNoticeDutyExcerpts(
    "How to apply the license. Copyright [yyyy] [copyright owner].",task).length,0);
  assert.equal(extractNoticeDutyExcerpts(apache,"Посчитай количество дней").length,0);
});

test("Claim Validator sees identified obligations from each attributed source",()=>{
  const context=buildClaimEvidenceContext({task,answer:"| Уведомления | Apache | BSD |",taskRunResult:{results:sources}});
  assert.equal(context.sourceCount,2);
  assert.match(context.sourceText,/PRIMARY SOURCE DUTY EXCERPTS/);
  assert.match(context.sourceText,/You must retain/);
  assert.match(context.sourceText,/Redistributions of source code must retain/);
  assert.match(context.sourceText,/SOURCE 2/);
});

test("Composer sees original clauses rather than the copyright template",()=>{
  const grid=buildComparisonEvidenceGrid(task,{results:sources});
  const row=grid.criteria.find(x=>x.id==="notices");
  assert.ok(row);
  assert.ok(row.excerpts[0].some(x=>x.includes("You must retain")));
  assert.ok(row.excerpts[1].some(x=>x.includes("must retain")));
});
