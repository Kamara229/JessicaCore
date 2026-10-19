import test from "node:test";
import assert from "node:assert/strict";
import {patchFocusedComparison} from "../core/composer/focusedComparisonPatch.js";
const original="| Критерий | Alpha | Beta |\n|---|---|---|\n| Уведомления и авторство | Неправильное условие | Требуется сохранять уведомления |";
const plan={
 previousAnswer:original,
 urls:["https://alpha.example/licensing","https://beta.example/licensing"],
 cells:[{id:1,criterion:"Уведомления и авторство",source:0}]
};
test("patch only rejected source-attributed cell",()=>{
 const r=patchFocusedComparison(plan,JSON.stringify({c:[[1,"Сохранять оригинальные уведомления об авторстве"]]}));
 assert.equal(r.ok,true);
 assert.match(r.text,/Сохранять оригинальные уведомления об авторстве/);
 assert.match(r.text,/Требуется сохранять уведомления/);
});
test("reject invalid evidence edits",()=>{
 assert.equal(patchFocusedComparison(plan,JSON.stringify({c:[[99,"Некорректный источник"]]})).ok,false);
});
