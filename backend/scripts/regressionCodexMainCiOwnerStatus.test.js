import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn} from "../core/codex/relayInterpreter.js";

const report="PR #58 объединён, CI в main успешен. "+
  "При сбое чтения владельца Candidate Proposal остаётся PROCESSING, "+
  "8 повторов без дублей; после восстановления — FAILED. "+
  "Далее — тест неверного owner receipt.";
const id="85ce513b-ed15-41b3-b831-4c0fcb1247c3";
const envelope=["JESSICA_REPLY_"+id,report,"JESSICA_END_"+id].join("\n");

test("real no-task PR handoff does not require inference provider",async()=>{
  for(const message of [report,envelope]){
    let invoked=0;
    const result=await interpretRelayTurn({message,history:[]},async()=>{
      invoked++;
      throw Error("inference outage");
    });
    assert.equal(invoked,0);
    assert.equal(result.action,"reply");
    assert.equal(result.task,"");
    assert.equal(result.continueThread,false);
    assert.match(result.reply,/GitHub, Render и Supabase/);
  }
});

test("malformed, disputed, and quoted statuses still require interpretation",async()=>{
  const bad=[
    report.replace("CI в main успешен","CI в main провален"),
    report.replace("объединён","предложен"),
    report.replace("Далее — тест неверного owner receipt",
      "Далее — выполнить действия на устройстве"),
    envelope.replace("JESSICA_END_","JESSICA_STOP_"),
    "Если "+report
  ];
  for(const message of bad){
    let invoked=0;
    const result=await interpretRelayTurn({message},async()=>{
      invoked++;
      throw Error("inference outage");
    });
    assert.equal(invoked,1);
    assert.equal(result.action,"reply");
    assert.equal(result.task,"");
  }
});
