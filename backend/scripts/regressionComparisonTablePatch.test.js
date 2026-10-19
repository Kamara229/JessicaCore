import test from "node:test";
import assert from "node:assert/strict";
import {buildComparisonCellMatrix} from "../core/validator/claim/comparisonCellMatrix.js";
import {freezeVerifiedComparisonCells} from "../core/composer/comparisonVerifiedCellLock.js";
import {mergeVerifiedComparisonRepair} from "../core/composer/comparisonTablePatch.js";

const urls=["https://www.apache.org/licenses/LICENSE-2.0.html",
 "https://opensource.org/license/bsd-3-clause"];
const answer=[
 "| Критерий | Apache 2.0 | BSD 3-Clause |",
 "|---|---|---|",
 "| Использование и изменения | Можно модифицировать | Использование зависит от уведомлений |",
 "| Распространение | Распространение разрешено | Распространение допустимо |",
 "| Уведомления и авторство | NOTICE в каждом файле | Сохранять копирайт |",
 "| Патентные условия | Патентный грант | Не удалось подтвердить по прочитанному тексту |",
 "| Ограничения | Нельзя использовать товарные знаки | Запрет продвижения именем |"
].join("\n");
const claims=buildComparisonCellMatrix(answer,urls).checks.map(c=>({
 sourceUrl:c.sourceUrl,claim:c.criterion+": "+c.value,
 supported:!((c.source===1&&c.criterion.startsWith("Использование"))
 ||(c.source===0&&c.criterion.startsWith("Уведомления")))
}));
const locks=freezeVerifiedComparisonCells(answer,claims,urls);

test("keep verified cells even if AI rewrites them inaccurately",()=>{
 const repaired=answer.replace("Можно модифицировать","Модификация запрещена")
  .replace("Использование зависит от уведомлений","Разрешается использование с изменениями")
  .replace("NOTICE в каждом файле","Обязан сохранить исходные уведомления")
  .replace("Нельзя использовать товарные знаки","Торговые знаки свободны");
 const merge=mergeVerifiedComparisonRepair(answer,repaired,locks,urls);
 assert.equal(merge.complete,true);
 assert.equal(merge.preserved,7);
 assert.ok(merge.patched>=2);
 assert.match(merge.text,/Можно модифицировать/);
 assert.match(merge.text,/Нельзя использовать товарные знаки/);
 assert.match(merge.text,/Разрешается использование с изменениями/);
 assert.match(merge.text,/Обязан сохранить исходные уведомления/);
});
test("unreadable rewritten columns cannot destroy original table",()=>{
 const repaired=answer.replace("Apache 2.0 | BSD 3-Clause","Первый | Второй");
 const merge=mergeVerifiedComparisonRepair(answer,repaired,locks,urls);
 assert.equal(merge.patched,0);
 assert.equal(merge.text,answer);
 assert.equal(merge.complete,true);
});
test("no source attribution fails closed",()=>{
 const repaired=answer.replace("NOTICE в каждом файле","Сохранить копирайт");
 const merge=mergeVerifiedComparisonRepair(answer,repaired,locks,[]);
 assert.equal(merge.complete,false);
 assert.equal(merge.text,answer);
});
test("unknown cells stay under independent follow-up validation",()=>{
 const revised=answer.replace(
  "Не удалось подтвердить по прочитанному тексту",
  "Не удалось подтвердить по тексту"
 );
 const merge=mergeVerifiedComparisonRepair(answer,revised,locks,urls);
 assert.match(merge.text,/Не удалось подтвердить по тексту/);
});
