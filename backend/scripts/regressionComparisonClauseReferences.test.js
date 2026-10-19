import test from "node:test";
import assert from "node:assert/strict";
import {buildComparisonClauseIndex} from "../core/validator/claim/comparisonClauseIndex.js";
import {canUseComparisonClauseIndex,resolveComparisonClauseReferences} from "../core/validator/claim/comparisonClauseReferences.js";
const urls=["https://docs.example.com/license-a","https://docs.example.com/license-b"];
const sources={results:[
 {tool:"web_fetch",success:true,data:{url:urls[0],content:"You may reproduce and distribute copies of the Work. You must retain all copyright and attribution notices in the Source form of any Derivative Works. If the Work includes a NOTICE text file, then any Derivative Works must include a readable copy of attribution notices. Each Contributor hereby grants to You a patent license. If You institute patent litigation, then patent licenses granted under this License shall terminate. This License does not grant permission to use the trade names of the Licensor."}},
 {tool:"web_fetch",success:true,data:{url:urls[1],content:"Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met. Redistributions of source code must retain the above copyright notice. Redistributions in binary form must reproduce the above copyright notice. Neither the name of the copyright holder nor the names of its contributors may be used to endorse products without written permission."}}
]};
const cells=[
 {id:1,source:0,sourceUrl:urls[0],criterion:"Уведомления",value:"Сохранить уведомления"},
 {id:2,source:1,sourceUrl:urls[1],criterion:"Использование и изменения",value:"Разрешается использовать"}
];
test("retrieve source-attributed operative clauses",()=>{
 const index=buildComparisonClauseIndex(sources);
 assert.ok(canUseComparisonClauseIndex({checks:cells},index));
 assert.ok(index.some(c=>c.source===0&&c.criterion==="notices"));
 assert.ok(index.some(c=>c.source===1&&c.criterion==="use"));
});
test("resolve compact IDs only for matching source and topic",()=>{
 const index=buildComparisonClauseIndex(sources);
 const a=index.find(c=>c.source===0&&c.criterion==="notices");
 const b=index.find(c=>c.source===1&&c.criterion==="use");
 const verified=resolveComparisonClauseReferences(JSON.stringify({c:[[1,[a.id]],[2,[b.id]]]}),cells,index);
 assert.equal(verified.valid,true);
 assert.equal(verified.c.length,2);
 assert.equal(verified.c[0][1][0],a.quote);
 assert.equal(resolveComparisonClauseReferences(JSON.stringify({c:[[1,[b.id]],[2,[a.id]]]}),cells,index).valid,false);
 assert.equal(resolveComparisonClauseReferences(JSON.stringify({c:[[1,[a.id]]]}),cells,index).valid,false);
});
