import test from "node:test";
import assert from "node:assert/strict";
import {buildComparisonCellMatrix} from "../core/validator/claim/comparisonCellMatrix.js";
import {locateComparisonCell} from "../core/validator/claim/tableCellLocator.js";
import {checkRequestedComparisonCoverage} from "../core/validator/claim/comparisonRequestedCoverage.js";
import {scaffoldComparisonTable} from "../core/composer/comparisonTableScaffold.js";

const urls=["https://www.apache.org/licenses/LICENSE-2.0.html","https://opensource.org/license/bsd-3-clause"];
const task="Сравни условия использования, распространения, сохранения уведомлений, патентные положения и ограничения Apache License 2.0 и BSD 3-Clause.";
const rows=[
  ["Использование и изменения","Apache License 2.0 разрешает копирование с условиями.","BSD 3-Clause разрешает изменение с условиями."],
  ["Распространение","Apache License 2.0 допускает распространение при соблюдении условий.","BSD 3-Clause допускает распространение при сохранении уведомлений."],
  ["Уведомления и авторство","Apache License 2.0 требует определённые уведомления.","BSD 3-Clause требует сохранять условия и отказ от гарантий."],
  ["Патентные условия","Apache License 2.0 предоставляет ограниченную по охвату патентную лицензию.","Не удалось подтвердить по прочитанному тексту."],
  ["Ограничения","Apache License 2.0 не предоставляет права на использование товарных знаков.","BSD 3-Clause ограничивает использование имён для продвижения."]
];
const answer=[
  "| Критерий | Apache License 2.0 | BSD 3-Clause |",
  "| --- | --- | --- |",
  ...rows.map(row=>"| "+row.join(" | ")+" |")
].join("\n");
const run={results:[
  {tool:"web_fetch",success:true,data:{url:urls[0],title:"Apache License 2.0",content:"Redistribution. You must retain copyright notices. Patent grants can terminate. "+rows[0][1]}},
  {tool:"web_fetch",success:true,data:{url:urls[1],title:"BSD 3-Clause",content:"Redistribution and use in source and binary forms are permitted provided that the following conditions are met. "+rows[0][2]}}
]};

test("source names inside factual cells do not hide requested criteria",()=>{
 const matrix=buildComparisonCellMatrix(answer,urls);
 assert.ok(matrix);
 assert.equal(matrix.totalRows,5);
 assert.equal(matrix.criteria.length,5);
 assert.equal(matrix.checks.length,9); // one explicit unverified patent condition
 assert.equal(checkRequestedComparisonCoverage(task,matrix).valid,true);
});

test("locate a comparison claim by its real row, not a source-named data cell",()=>{
 const found=locateComparisonCell({answer,sourceUrls:urls,claim:{claim:"Патентные условия",sourceUrl:urls[0]}});
 assert.ok(found);
 assert.equal(found.criterion,"Патентные условия");
 assert.ok(found.value.includes("патентную лицензию"));
});

test("scaffold sees one header, not a fake header in every source-attributed row",()=>{
 const result=scaffoldComparisonTable(task,run,answer);
 assert.equal(result.applied,false);
 assert.equal(result.reason,"complete");
 assert.equal(result.text,answer);
});

test("regression: ordinary complete table remains structurally stable",()=>{
 const generic=answer.replaceAll("Apache License 2.0 разрешает","Разрешено").replaceAll("BSD 3-Clause разрешает","Разрешено");
 const matrix=buildComparisonCellMatrix(generic,urls);
 assert.equal(matrix.totalRows,5);
 assert.equal(checkRequestedComparisonCoverage(task,matrix).valid,true);
});
