import test from "node:test";
import assert from "node:assert/strict";
import {partitionComparisonClaims} from "../core/validator/claim/claimUncertaintyScope.js";
import {locateComparisonCell} from "../core/validator/claim/tableCellLocator.js";

const apache="https://www.apache.org/licenses/LICENSE-2.0.html";
const bsd="https://opensource.org/license/bsd-3-clause";
const urls=[apache,bsd];
const answer=[
  "| Критерий | Apache 2.0 | BSD 3-Clause |",
  "| --- | --- | --- |",
  "| Использование | Разрешено | Разрешено |",
  "| Уведомления | Требуется сохранить уведомления | Сохранить copyright |",
  "| Патентные условия | Выдаётся патентная лицензия | Не удалось подтвердить по прочитанному тексту |",
  "| Ограничения | Нет разрешения на товарные знаки | Запрет на рекламное одобрение |"
].join("\n");

test("unverified cell is not a categorical absence of patent rights",()=>{
  const items=[
    {claim:"В тексте BSD 3-Clause отсутствует раздел о патентных правах",sourceUrl:bsd,supported:false},
    {claim:"BSD 3-Clause разрешает использование",sourceUrl:bsd,supported:true},
    {claim:"Apache 2.0 выдаёт патентную лицензию",sourceUrl:apache,supported:true}
  ];
  const result=partitionComparisonClaims(items,answer,urls);
  assert.equal(result.ignoredUncertainty,1);
  assert.deepEqual(result.claims,items.slice(1));
});

test("actually asserted negative stays subject to strict source validation",()=>{
  const explicit=answer.replace("Не удалось подтвердить по прочитанному тексту",
    "BSD 3-Clause не содержит положений о патентных правах");
  const item={claim:"BSD 3-Clause не содержит патентных положений",sourceUrl:bsd,supported:false};
  assert.equal(partitionComparisonClaims([item],explicit,urls).ignoredUncertainty,0);
});

test("categorical absence in prose cannot be silently discarded",()=>{
  const prose=answer+"\nBSD 3-Clause не содержит положений о патентных правах.";
  const claim={claim:"В BSD 3-Clause отсутствуют патентные положения",sourceUrl:bsd,supported:false};
  assert.equal(partitionComparisonClaims([claim],prose,urls).ignoredUncertainty,0);
});

test("a claimed Apache obligation is never hidden by unknown BSD patent cell",()=>{
  const claim={claim:"Apache требует сохранить copyright уведомления",sourceUrl:apache,supported:false};
  assert.equal(partitionComparisonClaims([claim],answer,urls).ignoredUncertainty,0);
});

test("reverse source order and alternate first-column headings remain mapped",()=>{
  const reversed=[
    "| Сравниваемые свойства | BSD 3-Clause | Apache 2.0 |",
    "| --- | --- | --- |",
    "| Патентные условия | Не удалось подтвердить | Выдаётся патентная лицензия |"
  ].join("\n");
  const claim={claim:"BSD 3-Clause не содержит патентных положений",sourceUrl:bsd,supported:false};
  const cell=locateComparisonCell({answer:reversed,sourceUrls:urls,claim});
  assert.equal(cell.column,1);
  assert.equal(partitionComparisonClaims([claim],reversed,urls).ignoredUncertainty,1);
});

test("unknown mapping does not authorize discarding any claim",()=>{
  const ambiguous="| Критерий | Первый | Второй |\n| --- | --- | --- |\n| Патенты | Да | Не удалось подтвердить |";
  const claim={claim:"BSD патентный грант не подтвержден",sourceUrl:bsd,supported:false};
  assert.equal(partitionComparisonClaims([claim],ambiguous,urls).ignoredUncertainty,0);
});
