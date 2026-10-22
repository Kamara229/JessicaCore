import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn, validateRelayDecision, explicitSameTestChat} from "../core/codex/relayInterpreter.js";
import {registerCodexRoutes} from "../http/routes/codexRoute.js";
import {recordCodexPhase,validateCodexPhase} from "../core/codex/relayTelemetry.js";

test("relay uses the plan and preceding results to understand natural chat", async () => {
    const result = await interpretRelayTurn({message:"Исправил backend. Продолжим контекст.",plan:"1. Язык. 2. Контекст.",
        history:[{role:"assistant",content:"Фактический ответ теста: ошибка"}]}, async (system,payload) => {
        assert.match(system,/Не утверждай, что тест выполнен/);
        assert.match(system,/НЕТ таких инструментов/);
        assert.equal(payload.plan,"1. Язык. 2. Контекст.");
        assert.match(payload.history[0].content,/ошибка/);
        return {action:"reply",reply:"Поняла исправление. Продолжим проверку контекста по плану.",task:""};
    });
    assert.equal(result.action,"reply");
});
test("only a verbatim test requested by ChatGPT is eligible for the ordinary chat", () => {
    const message = "В чате Jessica выполни: Сколько будет 2 + 2?";
    assert.equal(validateRelayDecision({action:"test",task:"Сколько будет 2 + 2?",continueThread:true},message).continueThread,false);
    assert.throws(()=>validateRelayDecision({action:"test",task:"Обнови GitHub"},message),/exact-quote/);
    assert.throws(()=>validateRelayDecision({action:"reply",reply:"OK",task:"лишняя задача"},message));
    assert.throws(()=>validateRelayDecision({action:"deploy",reply:"OK"},message));
});

test("continuing a Jessica test requires explicit same-chat instruction, never model flag alone", async () => {
    const task = "Какое число я назвал?";
    const decision = {action:"test",task,continueThread:true};
    assert.equal(validateRelayDecision(decision, "Выполни через Jessica: " + task).continueThread,false);
    const same = "Продолжи в том же чате Джессики. Задача: " + task;
    assert.equal(validateRelayDecision(decision,same).continueThread,true);
    const another = "Используй тот же тестовый чат Джессики. " + task;
    assert.equal(validateRelayDecision(decision,another).continueThread,true);
    assert.equal(validateRelayDecision(decision,"Do the next test in the same Jessica test chat: " + task).continueThread,true);
    assert.equal(validateRelayDecision({...decision,continueThread:false},same).continueThread,false);
    assert.equal(validateRelayDecision({action:"reply",reply:"Продолжим",continueThread:true},same).continueThread,false);
    assert.equal(explicitSameTestChat("Новый тест в новом чате. " + task),false);
    const interpreted = await interpretRelayTurn({message:"Новый независимый тест: " + task},
        async()=>decision);
    assert.equal(interpreted.continueThread,false);
});
test("ambiguous understanding fails instead of inventing continuation", async () => {
    await assert.rejects(interpretRelayTurn({message:"Готово"},async()=>"not json"));
    await assert.rejects(interpretRelayTurn({message:""},async()=>({action:"reply",reply:"OK"})));
});
test("Codex HTTP surface exposes interpretation and content-free diagnostics only", () => {
    const routes=[];
    registerCodexRoutes({post:(path,...handlers)=>routes.push({path,handlers})});
    assert.deepEqual(routes.map(r=>r.path),[
        "/api/codex/relay/phase",
        "/api/codex/relay/interpret"
    ]);
    for(const route of routes) assert.equal(route.handlers.length,2);
    assert.ok(routes.every(route=>!/(github|render|supabase|deploy|merge|edit|write)/i.test(route.path)));
});
test("Codex relay phases cannot send prompt, message, source or arbitrary labels to logs", () => {
    const id="f10c21ab-26e7-4451-826e-123456789012";
    const input={session:id,phase:"awaiting_markers",turn:1,elapsedSeconds:50,
        message:"PRIVATE_USER_MESSAGE",prompt:"PRIVATE_CODE",password:"SECRET"};
    const normalized=validateCodexPhase(input);
    assert.deepEqual(Object.keys(normalized),["session","phase","turn","elapsedSeconds"]);
    const emitted=[];
    assert.equal(recordCodexPhase(input,(label,value)=>emitted.push(JSON.stringify({label,value}))),true);
    assert.equal(emitted.length,1);
    assert.ok(!emitted[0].includes("PRIVATE"));
    assert.ok(!emitted[0].includes("SECRET"));
    assert.equal(validateCodexPhase({...input,phase:"editor_missing"}).phase,"editor_missing");
    assert.equal(validateCodexPhase({...input,phase:"send_control_missing"}).phase,"send_control_missing");
    assert.throws(()=>validateCodexPhase({...input,phase:"send_any_text"}));
    assert.throws(()=>validateCodexPhase({...input,turn:61}));
    assert.throws(()=>validateCodexPhase({...input,session:"my-device-id"}));
    assert.throws(()=>validateCodexPhase({...input,elapsedSeconds:-1}));
});
