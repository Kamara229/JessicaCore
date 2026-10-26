import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";
import {v2PersistenceHandoff}
 from "../core/codex/relayV2PersistenceHandoff.js";

const BODY="RPC v4: сохранение Skill с манифестом v2 подтверждено, транзакция откатана без изменений БД. CI пока не подтверждён. Следующий шаг — проверить восстановление v2 и отклонение повреждённых модулей.";
const ID="3cd82ffc-b8f1-423c-8927-a9356cf275b6";
const ENVELOPE="JESSICA_REPLY_"+ID+"\n\n"+BODY+"\n\nJESSICA_END_"+ID;

test("Codex interprets exact v2 rollback status without provider or test",async()=>{
 for(const message of [BODY,ENVELOPE]){
  let providerCalls=0;
  const result=await interpretRelayTurn({message},async()=>{
   providerCalls++;throw Error("external interpreter unavailable");
  });
  assert.equal(providerCalls,0);
  assert.equal(result.action,"reply");
  assert.equal(result.task,"");
  assert.equal(result.continueThread,false);
  assert.match(result.reply,/CI не подтверждён/);
  assert.match(result.reply,/Тест в обычном чате Jessica не запрошен/);
 }
});

test("v2 status handler refuses other tasks and broken markers",()=>{
 assert.equal(v2PersistenceHandoff(BODY.replace("откатана","сохранена")),null);
 assert.equal(v2PersistenceHandoff(BODY.replace("CI пока не подтверждён","CI успешно пройден")),null);
 assert.equal(v2PersistenceHandoff(BODY+" Выполни тест в Jessica."),null);
 assert.equal(v2PersistenceHandoff(ENVELOPE.replace("JESSICA_END_","JESSICA_END_00000000-")),null);
 assert.equal(v2PersistenceHandoff("Реши пример 2+2."),null);
});
