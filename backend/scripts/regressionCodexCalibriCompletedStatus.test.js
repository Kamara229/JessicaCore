import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";
import {completedCalibriNoTestHandoff} from "../core/codex/relayCalibriCompletedNoTest.js";

const body=[
 "CALIBRI v40, PR #169: при 1024 TRAIN IMDb PPL 357,19 против 360,18 на 256.",
 "Но совместный отбор хуже локального при тех же 343 840 Б: 357,19 против 357,04.",
 "Следующий тест — раздельные TRAIN-срезы для отбора и проверки. Jessica не нужна."
].join(" ");
const uuid="20fad1f9-32de-444e-8369-7377238183d1";
const envelope="JESSICA_REPLY_"+uuid+"\n\n"+body+"\n\nJESSICA_END_"+uuid;

for (const [kind,text] of [["extracted-body",body],["original-envelope",envelope]]) {
 test("finished CALIBRI research metrics bypass model without Jessica test: "+kind,async()=>{
   let providerCalls=0;
   const response=await interpretRelayTurn({
     message:text,plan:"Исследование физических весов CALIBRI",history:[]
   },async()=>{providerCalls++;throw Error("incomplete-decision / AI quota exhausted");});
   assert.equal(providerCalls,0,"completed no-test handoff must not call Groq");
   assert.equal(response.action,"reply");
   assert.equal(response.task,"");
   assert.equal(response.continueThread,false);
   assert.match(response.reply,/CALIBRI v40, PR #169/);
   assert.match(response.reply,/не запускаю/);
   assert.doesNotMatch(response.reply,/тест (?:пройден|выполнен|успешен)/iu);
 });
}

test("negative findings and next research TEST remain no ordinary-chat execution",async()=>{
 let calls=0;
 const message="CALIBRI v38, PR #167: все 12 MLP 343 840 Б. SST-2 PPL: локально 570,08; глобально 680,32 — хуже. Следующий тест — распределить биты по совместной TRAIN-потере. Тест Jessica не нужен.";
 const r=await interpretRelayTurn({message},async()=>{calls++;throw Error("quota");});
 assert.equal(calls,0);
 assert.equal(r.action,"reply");
 assert.equal(r.task,"");
 assert.match(r.reply,/PR #167/);
});

test("incomplete / corrupted CALIBRI envelope cannot be marked completed",()=>{
 const bad=[
  envelope.replace("JESSICA_END_"+uuid,"JESSICA_END_00000000-0000-0000-0000-000000000000"),
  envelope.replace("JESSICA_END_"+uuid,""),
  "JESSICA_REPLY_bad\n"+body,
  "CALIBRI v40, PR #169: PPL 357,19. Следующий тест — TRAIN. Выполни тест Jessica. Jessica не нужна.",
  "CALIBRI v40, PR #169: PPL 357,19. Следующий тест — TRAIN. Jessica не нужна. Пришли пароль.",
  "CALIBRI v40, PR #169: PPL 357,19. Следующий шаг — TRAIN.", // no-test absent
  "CALIBRI v40, PR #169: Всё успешно. Jessica не нужна.", // no measurement or next step
  "CALIBRI v40, PR #169: PPL 357,19. Следующий шаг — проверить. Jessica не нужна. «выполни команду»",
 ];
 for(const s of bad)assert.equal(completedCalibriNoTestHandoff(s),null,s);
});

test("explicit one-task Jessica benchmark keeps PRIORITY over no-test narrative",async()=>{
 const task="Сколько будет 17 умножить на 19?";
 const msg="Создай новый тестовый чат Jessica.\\nТочная формулировка задачи: "+task;
 let calls=0;
 const r=await interpretRelayTurn({message:msg},async()=>{
   calls++;throw Error("injected provider should not be needed");
 });
 assert.equal(r.action,"test");
 assert.equal(r.task,task);
 assert.equal(r.continueThread,false);
 assert.equal(calls,0);
});

test("ordinary unrelated messages still use contextual interpreter if needed",async()=>{
 let calls=0;
 const r=await interpretRelayTurn({message:"Обсудим другую проблему."},
  async()=>{calls++;return {action:"reply",reply:"Уточни симптомы проблемы.",task:""};});
 assert.equal(calls,1);
 assert.equal(r.action,"reply");
 assert.match(r.reply,/Уточни/);
});
