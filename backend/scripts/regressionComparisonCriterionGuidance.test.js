import test from "node:test";
import assert from "node:assert/strict";
import {comparisonCriterionGuidance} from "../core/composer/comparisonCriterionGuidance.js";
import {buildComparisonEvidenceGrid,renderComparisonEvidenceGrid} from "../core/composer/comparisonEvidenceGrid.js";

test("the different comparison rows stay semantically distinct",()=>{
 const text=comparisonCriterionGuidance([
  {id:"use"},{id:"distribution"},{id:"notices"},{id:"restrictions"},{id:"patents"}
 ]);
 assert.match(text,/УСЛОВИЯ|ИСПОЛЬЗОВАНИЕ/);
 assert.match(text,/Не пиши «без ограничений»/);
 assert.match(text,/Не добавляй отказ от гарантий/);
 assert.match(text,/Не превращай «если файл NOTICE имеется»/);
});
test("empty criteria produce no extra instructions",()=>{
 assert.equal(comparisonCriterionGuidance([]),"");
});
test("composer evidence map actually includes criterion boundaries",()=>{
 const run={results:[
  {tool:"web_fetch",success:true,data:{url:"https://a.example/licensing",
    title:"Document A",content:"Each Contributor hereby grants to You a non-exclusive copyright license. You may reproduce and distribute copies of the Work."}},
  {tool:"web_fetch",success:true,data:{url:"https://b.example/licensing",
    title:"Document B",content:"Redistribution and use in source and binary forms are permitted. Redistributions of source code must retain the above copyright notice."}}
 ]};
 const grid=buildComparisonEvidenceGrid("Сравни использование, распространение и уведомления",run);
 const rendered=renderComparisonEvidenceGrid(grid);
 assert.match(rendered,/РАЗГРАНИЧЕНИЕ КРИТЕРИЕВ/);
 assert.match(rendered,/ИСПОЛЬЗОВАНИЕ/);
 assert.match(rendered,/РАСПРОСТРАНЕНИЕ/);
});
