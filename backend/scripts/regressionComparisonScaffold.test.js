import test from "node:test";
import assert from "node:assert/strict";
import {scaffoldComparisonTable} from "../core/composer/comparisonTableScaffold.js";
import {buildComparisonCellMatrix} from "../core/validator/claim/comparisonCellMatrix.js";

const urls=["https://a.example/licenses/alpha","https://b.example/licenses/beta"];
const run={results:[
 {tool:"web_fetch",success:true,data:{url:urls[0],title:"Alpha",content:"Copyright license is granted to reproduce and distribute copies. You must retain the copyright notice."}},
 {tool:"web_fetch",success:true,data:{url:urls[1],title:"Beta",content:"Redistribution and use in source and binary forms are permitted. Redistributions of source code must retain the copyright notice."}}
]};
test("missing comparison dimensions are represented as unverified, not invented",()=>{
 const task="Сравни использование, распространение, сохранение уведомлений и ограничения";
 const answer="| Критерий | Alpha | Beta |\n|---|---|---|\n| Использование | Можно копировать | Можно использовать |";
 const updated=scaffoldComparisonTable(task,run,answer);
 assert.equal(updated.applied,true);
 assert.ok(updated.missing.length>=2);
 assert.ok(updated.text.includes("Не удалось подтвердить"));
 const matrix=buildComparisonCellMatrix(updated.text,urls);
 assert.ok(matrix&&matrix.totalRows>=3);
});
test("ambiguous source columns fail without guessing",()=>{
 const text="| Критерий | Первый | Второй |\n|---|---|---|\n| Использование | X | Y |";
 const result=scaffoldComparisonTable("Сравни использование",run,text);
 assert.equal(result.applied,false);
 assert.equal(result.text,text);
});
