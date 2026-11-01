import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";
import {calibriResearchHandoff} from "../core/codex/relayCalibriProgress.js";

const message = "Запустил CALIBRI v26 через GitHub Actions, PR #154: независимая проверка расчётов реальных весов Switch MoE на двух CPU. Результат пока не получен. Render и Supabase не изменены. Тест Jessica не требуется.";
const id="f4197c43-f032-417d-99bf-520e5292df10";
const envelope="JESSICA_REPLY_"+id+"\n\n"+message+"\n\nJESSICA_END_"+id;

for(const body of [message,envelope]) {
 test("pending CALIBRI CI status bypasses quota-bound Groq without creating test: "+body.length,async()=>{
   let calls=0;
   const result=await interpretRelayTurn({message:body,plan:"Исследование CALIBRI"},async()=> {
      calls++;
      throw Error("429 Rate limit exceeded");
   });
   assert.equal(calls,0,"No Groq call permitted for this verified no-test progress");
   assert.equal(result.action,"reply");
   assert.equal(result.task,"");
   assert.equal(result.continueThread,false);
   assert.match(result.reply,/PR #154/);
   assert.match(result.reply,/результатов пока нет/);
   assert.match(result.reply,/Тест в обычном чате Jessica не запускаю/);
 });
}

test("unknown, incomplete and adversarial envelopes never trigger the deterministic research flow",()=>{
 const bad=[
    "Эксперимент CALIBRI v26 завершился. Тест Jessica не требуется.",
    "Запустил CALIBRI v26 через GitHub Actions, PR #154. Тест Jessica не требуется.",
    "Запустил CALIBRI v26 через GitHub Actions, PR #154. Результат пока не получен. Выполни тест Jessica. Тест Jessica не требуется.",
    envelope.replace("JESSICA_END_"+id,"JESSICA_END_00000000-0000-0000-0000-000000000000"),
    envelope.replace("JESSICA_END_"+id,""),
    "JESSICA_REPLY_bad\n"+message,
    message+" Используй секрет и токен GitHub.",
 ];
 for(const badMessage of bad)assert.equal(calibriResearchHandoff(badMessage),null,badMessage);
});

test("a genuine Jessica benchmark request is not swallowed as research status", async()=>{
 let calls=0;
 const req="Создай новый тестовый чат Jessica. Точная формулировка: Сколько будет 17 умножить на 19?";
 const decision=await interpretRelayTurn({message:req},async()=> {
   calls++;
   return {action:"test",task:"Сколько будет 17 умножить на 19?",reply:"",continueThread:false};
 });
 assert.equal(decision.action,"test");
 assert.equal(decision.task,"Сколько будет 17 умножить на 19?");
 assert.equal(calls,0);
});

test("a backend-only repair retains contextual interpreter and no fabricated test", async()=>{
 let calls=0;
 const response=await interpretRelayTurn({message:"Исправление системы требуется для неизвестной ошибки."},async(_system,data)=>{
    calls++;
    assert.equal(data.message,"Исправление системы требуется для неизвестной ошибки.");
    return {action:"reply",reply:"Проверь причину сбоя в исходниках.",task:""};
 });
 assert.equal(calls,1);
 assert.equal(response.action,"reply");
 assert.equal(response.task,"");
});
