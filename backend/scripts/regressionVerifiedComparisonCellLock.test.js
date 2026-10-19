import test from "node:test";
import assert from "node:assert/strict";
import {buildComparisonCellMatrix} from "../core/validator/claim/comparisonCellMatrix.js";
import {freezeVerifiedComparisonCells,restoreVerifiedComparisonCells} from "../core/composer/comparisonVerifiedCellLock.js";

const urls=[
 "https://www.apache.org/licenses/LICENSE-2.0.html",
 "https://opensource.org/license/bsd-3-clause"
];
const answer=[
 "| Критерий | Apache 2.0 | BSD 3-Clause |",
 "| --- | --- | --- |",
 "| Использование и изменения | Можно создавать производные | Можно использовать без модификаций |",
 "| Распространение | Разрешено распространять копии | Допустима бинарная форма |",
 "| Уведомления и авторство | Не хватает цитаты | Требуется сохранять авторство |",
 "| Патентные условия | Предоставляются патентные права | Не удалось подтвердить по прочитанному тексту |",
 "| Ограничения | Не переданы права на торговые марки | Запрещена реклама именем правообладателя |"
].join("\n");
const claims=buildComparisonCellMatrix(answer,urls).checks.map(c=>({
 claim:c.criterion+": "+c.value,
 sourceUrl:c.sourceUrl,
 supported:c.source===1||c.criterion!=="Уведомления и авторство"
}));
test("verified comparison cells are frozen while one failed cell is editable",()=>{
 const frozen=freezeVerifiedComparisonCells(answer,claims,urls);
 assert.equal(frozen.length,8);
 const newAnswer=answer.replace("Можно создавать производные","Нет разрешения создавать производные")
  .replace("Не хватает цитаты","Возможно сохранять уведомления")
  .replace("Не переданы права на торговые марки","Разрешены любые товарные знаки");
 const updated=restoreVerifiedComparisonCells(newAnswer,frozen,urls);
 assert.equal(updated.complete,true);
 assert.equal(updated.restored,8);
 assert.match(updated.text,/Можно создавать производные/);
 assert.match(updated.text,/Не переданы права на торговые марки/);
 assert.match(updated.text,/Возможно сохранять уведомления/);
});
test("a repair cannot drop previously validated comparison rows",()=>{
 const frozen=freezeVerifiedComparisonCells(answer,claims,urls);
 const changed=answer.split("\n").filter(line=>!line.startsWith("| Ограничения")).join("\n");
 assert.equal(restoreVerifiedComparisonCells(changed,frozen,urls).complete,false);
});
test("source columns must be mapped without guessing",()=>{
 const frozen=freezeVerifiedComparisonCells(answer,claims,urls);
 const changed=answer.replace("Apache 2.0 | BSD 3-Clause","Первый | Второй");
 assert.equal(restoreVerifiedComparisonCells(changed,frozen,urls).complete,false);
});
test("unverified or unsupported cells are never locked",()=>{
 const unknown=claims.map(c=>({...c,supported:false}));
 assert.deepEqual(freezeVerifiedComparisonCells(answer,unknown,urls),[]);
});
