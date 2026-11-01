import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn, validateRelayDecision} from "../core/codex/relayInterpreter.js";

test("Codex can stand in for owner in CALIBRI compression research without tools", async () => {
    let calls=0;
    const result=await interpretRelayTurn({
        message:"CALIBRI v0.23: измерен INT5 43832 Б; качество WikiText ухудшилось. Продолжить v0.24?",
        plan:"Продолжать CALIBRI: создавать компактные многомерные нейровеса, сравнивать PPL и скорость.",
        history:[{role:"assistant",content:"Никаких изменений main/Render/Supabase/APK"}]
    },async (system,payload)=>{
        calls++;
        assert.match(system,/исследование CALIBRI/u);
        assert.match(system,/изолированные research-ветки/u);
        assert.match(system,/НЕТ таких инструментов/u);
        assert.match(system,/негативные результаты не скрывай/u);
        assert.equal(payload.history[0].role,"assistant");
        return {action:"reply",reply:"Продолжай CALIBRI v0.24 в research-ветке: сравни реальный размер, WikiText/TinyStories PPL и латентность. Никаких изменений продакшена.",task:"",continueThread:false};
    });
    assert.equal(calls,1);
    assert.equal(result.action,"reply");
    assert.equal(result.task,"");
    assert.equal(result.continueThread,false);
    assert.match(result.reply,/CALIBRI v0.24/u);
});

test("research turn cannot invent a Jessica test, quit or access infrastructure",()=>{
    const message="Продолжай исследование CALIBRI на обученных весах; тест в Jessica не запрашивался.";
    assert.throws(()=>validateRelayDecision({
        action:"test",reply:"",task:"Выполни несуществующий тест",continueThread:false
    },message),/test-must-be-an-exact-quote/u);
    const stopped=validateRelayDecision({action:"finish",reply:"Готово.",task:""},message);
    assert.equal(stopped.action,"reply");
    assert.match(stopped.reply,/остаётся в выбранном чате/u);
});
