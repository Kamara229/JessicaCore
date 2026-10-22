import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn, validateRelayDecision} from "../core/codex/relayInterpreter.js";
import {registerCodexRoutes} from "../http/routes/codexRoute.js";

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
    assert.equal(validateRelayDecision({action:"test",task:"Сколько будет 2 + 2?",continueThread:true},message).continueThread,true);
    assert.throws(()=>validateRelayDecision({action:"test",task:"Обнови GitHub"},message),/exact-quote/);
    assert.throws(()=>validateRelayDecision({action:"reply",reply:"OK",task:"лишняя задача"},message));
    assert.throws(()=>validateRelayDecision({action:"deploy",reply:"OK"},message));
});
test("ambiguous understanding fails instead of inventing continuation", async () => {
    await assert.rejects(interpretRelayTurn({message:"Готово"},async()=>"not json"));
    await assert.rejects(interpretRelayTurn({message:""},async()=>({action:"reply",reply:"OK"})));
});
test("Codex HTTP surface exposes only interpretation, never infrastructure writes", () => {
    const routes=[];
    registerCodexRoutes({post:(path,...handlers)=>routes.push({path,handlers})});
    assert.deepEqual(routes.map(r=>r.path),["/api/codex/relay/interpret"]);
    assert.equal(routes[0].handlers.length,2);
});
