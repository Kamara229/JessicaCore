import test from "node:test";
import assert from "node:assert/strict";
import {findAvoidableUnknownComparisonCell} from "../core/validator/claim/comparisonAvoidableUnknown.js";
import {indexExplicitComparisonTopics} from "../core/validator/claim/comparisonUnknownProofIndex.js";

const urls=["https://a.example/license/alpha","https://b.example/license/beta"];
const result={results:[
 {tool:"web_fetch",success:true,data:{url:urls[0],content:
 "Each Contributor hereby grants to You a non-exclusive copyright license. "+
 "You may reproduce and distribute copies of the Work. "+
 "You must retain all copyright and attribution notices from the Source form. "+
 "This License does not grant permission to use the trade names of the Licensor."}},
 {tool:"web_fetch",success:true,data:{url:urls[1],content:
 "Redistribution and use in source and binary forms are permitted provided that the following conditions are met. "+
 "Redistributions of source code must retain the above copyright notice. "+
 "Neither the name of the copyright holder nor the names of its contributors may be used to endorse products."}}
]};
const task="Сравни использование, распространение, уведомления, патентные условия и ограничения";
const UNKNOWN="Не удалось подтвердить по прочитанному тексту";
function table(topic){
 return "| Критерий | Alpha | Beta |\n| --- | --- | --- |\n| "+topic+" | "+UNKNOWN+" | "+UNKNOWN+" |";
}
test("positive source provisions prevent a false complete status",()=>{
 const index=indexExplicitComparisonTopics(result.results);
 for(const source of [0,1])for(const topic of ["use","distribution","notices","restrictions"])
  assert.equal(index.get(source)?.has(topic),true);
 for(const topic of ["Использование","Распространение","Уведомления","Ограничения"])
  assert.ok(findAvoidableUnknownComparisonCell(task,table(topic),result.results));
});
test("lack of indexed patent language never establishes patent absence",()=>{
 assert.equal(findAvoidableUnknownComparisonCell(task,table("Патентные условия"),result.results),null);
});
