import test from "node:test";
import assert from "node:assert/strict";
import {buildComparisonEvidenceGrid,renderComparisonEvidenceGrid}
  from "../core/composer/comparisonEvidenceGrid.js";
import {buildComposerContext}
  from "../core/composer/composerContext.js";

const task="Сравни две лицензии Apache License 2.0 и BSD 3-Clause: использование, распространение, уведомления, патентные положения и ограничения.";
const apache="Apache License Version 2.0. "+("Introduction copyright and use. ".repeat(35))+
    "Grant of Patent License. Each contributor grants perpetual worldwide patent license. "+
    ("Copies may be distributed in source and binary forms. ".repeat(25))+
    "If you institute patent litigation, licenses for that Work shall terminate. "+
    ("Copyright notice required with redistribution. ".repeat(21))+
    "The License does not grant permission to use trade names, trademarks or product names.";
const bsd="BSD 3-Clause License. Redistribution and use in source and binary forms, with or without modification, are permitted. "+
    ("Redistributions of source code must retain the above copyright notice. ".repeat(13))+
    "Neither the name of the copyright holder nor the names of its contributors may be used to endorse products.";
const run={results:[
    {tool:"web_fetch",success:true,data:{url:"https://www.apache.org/licenses/LICENSE-2.0.html",content:apache}},
    {tool:"web_fetch",success:true,data:{url:"https://opensource.org/license/bsd-3-clause",content:bsd}}
]};
test("evidence-first grid has both origins and all expressly requested criteria",()=>{
    const grid=buildComparisonEvidenceGrid(task,run);
    assert.deepEqual(grid.sources.map(x=>x.id),["S1","S2"]);
    assert.deepEqual(grid.criteria.map(x=>x.id),["use","distribution","notices","patents","restrictions"]);
    assert.equal(grid.criteria.find(x=>x.id==="patents").excerpts[1].length,0,
       "BSD unrelated grant/permission boilerplate must not count as patent evidence");
    const c=renderComparisonEvidenceGrid(grid);
    assert.match(c,/ИСТОЧНИК 1/);
    assert.match(c,/ИСТОЧНИК 2/);
    assert.match(c,/Grant of Patent License/);
    assert.match(c,/patent litigation/);
    assert.match(c,/Не удалось найти подтверждающую выдержку по этому критерию/);
    assert.ok(c.length<9000);
    assert.match(buildComposerContext(task,{evidence:{mode:"source_content"}},run),/КАРТА ПЕРВОИСТОЧНИКОВ/);
});
test("grid rejects truncated or single-source fetch results",()=>{
    assert.equal(buildComparisonEvidenceGrid(task,{results:run.results.slice(0,1)}),null);
    assert.equal(buildComparisonEvidenceGrid(task,{results:[
       run.results[0],{...run.results[1],data:{...run.results[1].data,truncated:true}}
    ]}),null);
});
test("grid is evidence extraction, not an unsupported patent absence conclusion",()=>{
    const grid=buildComparisonEvidenceGrid(task,run);
    const rendered=renderComparisonEvidenceGrid(grid);
    assert.doesNotMatch(rendered,/BSD\s*3-Clause\s*(?:не содержит|без патентов)/i);
    assert.match(rendered,/Нет цитаты по критерию/);
});
