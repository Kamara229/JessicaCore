import test from "node:test";
import assert from "node:assert/strict";
import {resolveComparisonClauseReferences} from "../core/validator/claim/comparisonClauseReferences.js";
import {recoverMisattributedComparisonReferences} from "../core/validator/claim/comparisonCitationRecovery.js";

const cells=[
 {id:1,source:0,criterion:"Распространение",value:"Условное распространение."},
 {id:2,source:1,criterion:"Распространение",value:"Условное распространение."}
];
const quoteA="You may reproduce and distribute copies of the Work provided the license conditions are met.";
const quoteB="Redistributions of source code must retain the copyright notice and list of license conditions.";
const index=[
 {id:"E4",source:0,criterion:"distribution",quote:quoteA},
 {id:"E6",source:1,criterion:"distribution",quote:quoteB}
];

test("source mismatch cannot certify a claim",()=>{
 const result=resolveComparisonClauseReferences(
  JSON.stringify({c:[[1,["E4"]],[2,["E4"]]]}),cells,index);
 assert.equal(result.valid,true);
 assert.deepEqual(result.c,[[1,[quoteA]],[2,[]]]);
 assert.deepEqual(result.misattributedIds,[2]);
});

test("a source-isolated second verification may recover only the bad citation",async()=>{
 let calls=0;
 const result=await recoverMisattributedComparisonReferences({
  matrix:{checks:cells},index,misattributedIds:[2],task:"Сравни",
  runBatch:async args=>{
   calls++;
   assert.deepEqual(args.route.checks.map(c=>c.id),[2]);
   assert.deepEqual(args.index.map(e=>e.id),["E6"]);
   return {valid:true,pairs:[[2,[quoteB]]]};
  }
 });
 assert.equal(calls,1);
 assert.deepEqual(result.restored,[[2,[quoteB]]]);
});

test("unsupported citations remain unsupported",async()=>{
 const result=await recoverMisattributedComparisonReferences({
  matrix:{checks:cells},index,misattributedIds:[2],task:"Сравни",
  runBatch:async()=>({valid:true,pairs:[[2,[]]]})
 });
 assert.deepEqual(result.restored,[]);
});

test("ordinary empty references are not automatically retried",async()=>{
 const result=resolveComparisonClauseReferences(
  JSON.stringify({c:[[1,["E4"]],[2,[]]]}),cells,index);
 assert.deepEqual(result.misattributedIds,[]);
});
