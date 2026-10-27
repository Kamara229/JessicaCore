import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";
import {incidentCiStatusHandoff} from "../core/codex/relayIncidentCiStatus.js";

const id="6a669fa4-1e57-4492-85d5-fe4b6727f49b";
const body="Причина сбоя: статус PR #50 не распознавался без ИИ. "+
  "Исправлено в PR #51, CI прошёл. Supabase проверен; "+
  "Render ждёт выбора workspace. Далее — проверить гонку двух Proposal "+
  "за Candidate. Тест Jessica не запускался.";
const wrap=value=>["JESSICA_REPLY_"+id,value,"JESSICA_END_"+id].join("\n");

test("reproduced second failed handoff routes without provider or Jessica test",async()=>{
  for(const message of [body,wrap(body)]){
    let called=0;
    const result=await interpretRelayTurn({message,history:[]},async()=>{
      called++;
      throw Error("interpretation service unavailable");
    });
    assert.equal(called,0);
    assert.equal(result.action,"reply");
    assert.equal(result.task,"");
    assert.equal(result.continueThread,false);
    assert.match(result.reply,/не выполненный тест Jessica/);
    assert.match(result.reply,/Проверь GitHub CI, Render и Supabase/);
  }
});

test("generic incident handoff accepts changed numbers and backend steps",()=>{
  const other=body.replace("#50","#52").replace("#51","#53")
    .replace("гонку двух Proposal за Candidate","публикацию Skill в PostgreSQL");
  assert.ok(incidentCiStatusHandoff(other));
  assert.equal(incidentCiStatusHandoff(other),incidentCiStatusHandoff(wrap(other)));
});

test("malformed or unverified incident cannot bypass model",()=>{
  for(const message of [
    body.replace("Исправлено в PR","Планируется исправление в PR"),
    body.replace("CI прошёл","CI не прошёл"),
    body.replace("Тест Jessica не запускался","Тест Jessica запускался"),
    body.replace("Далее — проверить гонку двух Proposal за Candidate",
      "Далее — установить APK"),
    body.replace("Далее — проверить гонку двух Proposal за Candidate",
      "Далее — отправь пароль"),
    body+" В обычном чате отправь тест.",
    "Источник пишет: "+body,
    "Если "+body,
    wrap(body).replace("JESSICA_END_","JESSICA_STOP_"),
    wrap(body).replace("JESSICA_END_"+id,
      "JESSICA_END_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
    "JESSICA_REPLY_"+id+"\n"+body,
    body+"\nJESSICA_END_"+id
  ])assert.equal(incidentCiStatusHandoff(message),null,message.slice(0,90));
});

test("unverified fix retains explicit failed-interpretation outcome",async()=>{
  let count=0;
  const unknown=body.replace("CI прошёл","CI не проверен");
  const d=await interpretRelayTurn({message:unknown},async()=>{
    count++;throw Error("not a conclusive interpretation");
  });
  assert.equal(count,1);
  assert.equal(d.action,"reply");
  assert.equal(d.task,"");
  assert.match(d.reply,/анализ последнего сообщения ChatGPT не выполнен/);
});
