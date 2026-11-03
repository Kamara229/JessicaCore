import test from "node:test";
import assert from "node:assert/strict";
import {codexIncidentNoTestStatus} from "../core/codex/relayIncidentNoTestStatus.js";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";

const message="Причина: backend не распознавал формат «Тест: …» и обращался к AI-интерпретатору. Исправление в PR #220, backend CI пройден. Render и Supabase проверены. Live не менял. Тест Jessica не запускался.";
const id="928119e0-77a9-4d05-a9fd-22e66963e6c8";
const wrapped=["JESSICA_REPLY_"+id,message,"JESSICA_END_"+id].join("\n");

test("exact previous ChatGPT no-test backend incident avoids unavailable AI interpreter",async()=>{
  for(const text of [message,wrapped]){
    let calls=0;
    const decision=await interpretRelayTurn({message:text},async()=>{
      calls++;
      throw Error("provider 429 unavailable");
    });
    assert.equal(calls,0,"no provider call for exact known no-test status");
    assert.equal(decision.action,"reply");
    assert.equal(decision.task,"");
    assert.equal(decision.continueThread,false);
    assert.match(decision.reply,/не запускаю/u);
    assert.match(decision.reply,/исправлени[ея]/u);
    assert.doesNotMatch(decision.reply,/тест выполнен|успешный тест/iu);
  }
});
test("quoted example of a test is not executed as actual Jessica test",()=>{
  const reply=codexIncidentNoTestStatus(wrapped);
  assert.ok(reply);
  assert.match(reply,/результата нет/u);
});
test("malformed envelopes, omitted no-test boundary, and new commands fail closed",()=>{
  for(const body of [
    message.replace("Тест Jessica не запускался.","Тест Jessica был успешен."),
    message.replace("Live не менял.","Live обновлён."),
    message.replace("backend CI пройден.","backend CI не пройден."),
    message.replace("PR #220","PR #"),
    message+" Отправь в обычный чат Jessica новый тест.",
    "Следующий тест в обычном чате Jessica: «Рассчитай число»",
    "Причина: backend не распознавал формат «Тест: …». Пожалуйста, установи APK."
  ])assert.equal(codexIncidentNoTestStatus(body),null,body);
  for(const text of [
    "JESSICA_REPLY_"+id+"\n"+message+"\nJESSICA_END_00000000-0000-0000-0000-000000000000",
    "JESSICA_REPLY_"+id+"\n"+message,
    "JESSICA_REPLY_"+id+"\n"+message+"\n"+message+"\nJESSICA_END_"+id,
    "JESSICA_REPLY_"+id+"\n"+message+"\nJESSICA_END_"+id+"\nEXTRA",
  ])assert.equal(codexIncidentNoTestStatus(text),null,text);
});
