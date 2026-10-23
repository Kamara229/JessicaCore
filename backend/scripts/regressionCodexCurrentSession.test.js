import test from "node:test";
import assert from "node:assert/strict";
import {directCurrentChatReply} from "../core/codex/relayCurrentChatPolicy.js";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";

test("same session imperative has priority over generic missing-test fallback",async()=>{
  const message=[
    "Не создавай новый тестовый чат Jessica.",
    "Продолжай в этой же сессии ChatGPT.",
    "Самостоятельно отправь следующее сообщение:",
    "«Кодекс: ответ принят. Продолжаю диалог»"
  ].join("\n");
  assert.equal(directCurrentChatReply(message),"Кодекс: ответ принят. Продолжаю диалог");
  let calls=0;
  const r=await interpretRelayTurn({message},async()=>{calls++;throw Error("Unexpected model");});
  assert.equal(calls,0);
  assert.equal(r.action,"reply");
  assert.equal(r.reply,"Кодекс: ответ принят. Продолжаю диалог");
});
