import test from "node:test";
import assert from "node:assert/strict";
import {guardComparisonUnknowns} from "../core/composer/comparisonUnknownGuard.js";

const urls=["https://www.apache.org/licenses/LICENSE-2.0.html","https://opensource.org/license/bsd-3-clause"];
const plan={evidence:{mode:"source_content"}};
const run={results:[
  {tool:"web_fetch",success:true,data:{url:urls[0],title:"Apache License 2.0",content:"Apache License 2.0. Grant of Patent License. Subject to terms, each Contributor hereby grants a patent license. Patent licenses shall terminate after specified litigation."}},
  {tool:"web_fetch",success:true,data:{url:urls[1],title:"BSD 3-Clause",content:"Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met. Redistributions of source code must retain the copyright notice."}}
]};
const answer=[
  "| Критерий | Apache License 2.0 | BSD 3-Clause |",
  "| --- | --- | --- |",
  "| Патентные условия | Apache 2.0 предусматривает патентную лицензию при соблюдении условий. | Не удалось найти подтверждающую выдержку в тексте BSD 3-Clause (отсутствует). |"
].join("\n");

test("an unknown patent cell must not imply absence in the primary source",()=>{
 const result=guardComparisonUnknowns("Сравни патентные условия",plan,run,answer);
 assert.equal(result.changes,1);
 assert.match(result.text,/Не удалось подтвердить по прочитанному тексту/);
 assert.doesNotMatch(result.text,/\(отсутствует\)/);
 assert.match(result.text,/Apache 2.0 предусматривает патентную лицензию/);
});

test("a correctly qualified unknown remains unchanged",()=>{
 const text=answer.replace("Не удалось найти подтверждающую выдержку в тексте BSD 3-Clause (отсутствует).","Не удалось подтвердить по прочитанному тексту");
 const result=guardComparisonUnknowns("Сравни патентные условия",plan,run,text);
 assert.equal(result.changes,0);
 assert.equal(result.text,text);
});
