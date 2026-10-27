import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn}
  from "../core/codex/relayInterpreter.js";
import {mergedPrProgressHandoff}
  from "../core/codex/relayMergedPrProgress.js";

const id="cf4898d7-93a0-4f3c-a5a3-388f550b3d69";
const body="PR #50 объединён, CI успешен. Потеря ACK после PROMOTED при двух "+
  "worker проверена, дублей нет. Supabase без изменений. "+
  "Следующий шаг — проверить гонку двух разных Proposal за одним Candidate.";
const wrap=value=>["JESSICA_REPLY_"+id,value,"JESSICA_END_"+id].join("\n");

test("actual failed Codex turn is interpreted without Groq (body and envelope)",async()=>{
  for(const message of [body,wrap(body)]){
    let calls=0;
    const result=await interpretRelayTurn({message,history:[]},async()=>{
      calls++;
      throw Error("interpretation provider failed");
    });
    assert.equal(calls,0,"a safe confirmed no-test status must not need the provider");
    assert.equal(result.action,"reply");
    assert.equal(result.task,"");
    assert.equal(result.continueThread,false);
    assert.match(result.reply,/Проверь статус через GitHub, Render и Supabase/);
    assert.match(result.reply,/тест Jessica не выполнялся/);
  }
});

test("next backend step works regardless of PR number",()=>{
  const valid=body.replace("#50","#51")
    .replace("проверить гонку двух разных Proposal за одним Candidate",
      "проверить атомарную запись Skill в PostgreSQL");
  assert.ok(mergedPrProgressHandoff(valid));
});

test("incomplete, unsafe or unverifiable status cannot bypass provider",()=>{
  const failures=[
    body.replace("объединён","предложен"),
    body.replace("CI успешен","CI не успешен"),
    body.replace("Следующий шаг — проверить","Возможно, следующим шагом проверить"),
    body.replace("проверить гонку двух разных Proposal за одним Candidate",
      "установить APK"),
    body.replace("проверить гонку двух разных Proposal за одним Candidate",
      "отправь пароль"),
    body+" В обычном чате отправь тест.",
    "В цитате: "+body,
    "Если PR #50 объединён, CI успешен. "+body,
    wrap(body).replace("JESSICA_END_","JESSICA_STOP_"),
    wrap(body).replace("JESSICA_END_"+id,"JESSICA_END_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
    "JESSICA_REPLY_"+id+"\n"+body,
    body+"\nJESSICA_END_"+id
  ];
  for(const message of failures){
    assert.equal(mergedPrProgressHandoff(message),null,message.slice(0,90));
  }
});

test("failed provider remains a visible error for ambiguous status",async()=>{
  let called=0;
  const ambiguous=body.replace("CI успешен","CI ещё не проверен");
  const result=await interpretRelayTurn({message:ambiguous},async()=>{
    called++;
    throw Error("ambiguous context");
  });
  assert.equal(called,1);
  assert.equal(result.action,"reply");
  assert.match(result.reply,/анализ последнего сообщения ChatGPT не выполнен/);
  assert.equal(result.task,"");
});
