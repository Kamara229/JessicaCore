import test from "node:test";
import assert from "node:assert/strict";
import {buildComparisonRepairHints} from "../core/composer/comparisonRepairHints.js";
import {buildComposerContext} from "../core/composer/composerContext.js";
const task="Сравни лицензии A и B по использованию, распространению и уведомлениям; покажи таблицу.";
const run={results:[
  {tool:"web_fetch",success:true,data:{url:"https://example.org/license/a",title:"License A",content:"You may reproduce and distribute copies. You must retain copyright notices."}},
  {tool:"web_fetch",success:true,data:{url:"https://example.org/license/b",title:"License B",content:"Redistribution and use in source and binary forms are permitted. Redistributions of source code must retain the above copyright notice."}}
]};
const plan={evidence:{mode:"source_content"}};
test("comparison-specific repair advice is not added without failure",()=>{
 assert.equal(buildComparisonRepairHints(""),"");
});
test("failed comparison receives separate substantive repair guidance",()=>{
 const result=buildComposerContext(task,plan,run,{
   previousAnswer:"| Критерий | A | B |",
   validatorReason:"Распространение: цитата не подтверждает юридический смысл"});
 assert.match(result,/ПРИЧИНА ОТКЛОНЕНИЯ/);
 assert.match(result,/РЕЖИМ ИСПРАВЛЕНИЯ СРАВНЕНИЯ/);
 assert.match(result,/Не смешивай разрешение и обязанность/);
});
test("fresh comparison does not receive repair instructions",()=>{
 const result=buildComposerContext(task,plan,run);
 assert.doesNotMatch(result,/РЕЖИМ ИСПРАВЛЕНИЯ СРАВНЕНИЯ/);
});
