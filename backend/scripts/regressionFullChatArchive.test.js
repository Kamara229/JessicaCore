import test from "node:test";
import assert from "node:assert/strict";
import {validateChatArchive} from "../storage/chatDebugArchive.js";
import {fallbackTitle} from "../core/chat/chatTitleService.js";

const first="11111111-1111-4111-8111-111111111111";
const second="22222222-2222-4222-8222-222222222222";
const msg=(text)=>({text,isUser:true,timestamp:1000});
const chat=(id,title,text)=>({id,title,updatedAt:1000,messages:[msg(text)]});

test("full diagnostic archive retains every conversation and message",()=>{
  const snapshot=validateChatArchive({threads:[
    chat(first,"Финансы","Проверь баланс"),
    chat(second,"Лицензии","Сравни два текста")
  ],activeThreadId:second});
  assert.equal(snapshot.valid,true);
  assert.equal(snapshot.row.thread_count,2);
  assert.equal(snapshot.row.message_count,2);
  assert.equal(snapshot.row.active_thread_id,second);
  assert.equal(snapshot.row.threads[0].messages[0].text,"Проверь баланс");
});
test("deleted chat is absent from the next complete snapshot",()=>{
  const old=validateChatArchive({threads:[chat(first,"A","a"),chat(second,"B","b")]});
  const next=validateChatArchive({threads:[chat(second,"B","b")],activeThreadId:second});
  assert.equal(old.row.threads.length,2);
  assert.deepEqual(next.row.threads.map(x=>x.id),[second]);
});
test("reject duplicate IDs, unsafe paths and unsupported oversized messages",()=>{
  assert.equal(validateChatArchive({threads:[chat(first,"A","a"),chat(first,"B","b")]}).valid,false);
  assert.equal(validateChatArchive({threads:[chat("../secret","A","a")]}).valid,false);
  assert.equal(validateChatArchive({threads:[chat(first,"A","x".repeat(20001))]}).valid,false);
});
test("offline title fallback is short and nonempty",()=>{
  assert.ok(fallbackTitle("Сравни лицензии Apache и BSD").length>=3);
  assert.ok(fallbackTitle("Сравни лицензии Apache и BSD").length<=64);
});
