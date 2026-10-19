import test from "node:test";
import assert from "node:assert/strict";
import {buildChatDebugSnapshot, saveChatDebugSnapshot} from "../storage/chatDebugSnapshot.js";

test("private mirror shows eight most recent on-device messages plus exact request and reply",()=>{
  const conversation=Array.from({length:14},(_,index)=>({
    role:index%2===0?"user":"assistant", content:"Сообщение "+index
  }));
  const task="Сравни лицензии\nпо официальным ссылкам.";
  const answer="| Критерий | Apache | BSD |\n| --- | --- | --- |\n| Патенты | Есть | Не проверено |";
  const s=buildChatDebugSnapshot({
    task,conversation,
    attachments:[{name:"reference.pdf",text:"Sensitive attached contents must never be stored"}],
    result:{text:answer,success:false,internalData:{secret:"Nope"}}
  });
  assert.equal(s.messages.length,10);
  assert.equal(s.messages[0].text,"Сообщение 6");
  assert.deepEqual(s.messages.slice(-2).map(m=>m.role),["user","assistant"]);
  assert.equal(s.messages[8].text,task);
  assert.deepEqual(s.messages[8].attachments,["reference.pdf"]);
  assert.equal(s.messages[9].text,answer);
  assert.equal(s.result_success,false);
  assert.ok(!JSON.stringify(s).includes("Sensitive attached contents"));
  assert.ok(!JSON.stringify(s).includes("Nope"));
});

test("disabled diagnostic never accesses the network",async()=>{
  const old=process.env.JESSICA_CHAT_DEBUG_ENABLED;
  try {
    process.env.JESSICA_CHAT_DEBUG_ENABLED="false";
    assert.deepEqual(await saveChatDebugSnapshot({}),{saved:false,reason:"disabled"});
  } finally {
    if(old===undefined) delete process.env.JESSICA_CHAT_DEBUG_ENABLED;
    else process.env.JESSICA_CHAT_DEBUG_ENABLED=old;
  }
});
