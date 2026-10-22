// Live release smoke check. Uses the existing app secret only with Jessica's fixed HTTPS endpoint.
// No ChatGPT sends, test execution, repository changes or deployment operations.
import assert from "node:assert/strict";
import {setTimeout as delay} from "node:timers/promises";

const token = process.env.JESSICA_APP_TOKEN;
if (!token) throw Error("JESSICA_APP_TOKEN is required for the live relay smoke check");
const endpoint = "https://jessicacore.onrender.com/api/codex/relay/interpret";

async function interpret(payload, waitForDeploy = false) {
    const deadline = Date.now() + (waitForDeploy ? 8 * 60 * 1000 : 0);
    while (true) {
        let response;
        try {
            response = await fetch(endpoint, {
                method:"POST", redirect:"error", signal:AbortSignal.timeout(90000),
                headers:{"Content-Type":"application/json", "X-Jessica-Token":token},
                body:JSON.stringify(payload)
            });
        } catch (error) {
            if (Date.now() >= deadline) throw Error("Jessica endpoint unavailable", {cause:error});
            await delay(10000); continue;
        }
        const data = await response.json().catch(()=>({}));
        if (response.ok && data.success) return data;
        if (data.reason === "relay-understanding-unavailable") throw Error("Live interpreter could not produce a valid decision");
        if (![404, 502, 503, 504].includes(response.status) || Date.now() >= deadline)
            throw Error(`Relay smoke check failed: HTTP ${response.status}`);
        await delay(10000);
    }
}

const acknowledgement = await interpret({
    plan:"Улучшить обработку контекста, затем обсудить следующий шаг.", history:[],
    message:"Backend обновлён. Сейчас тесты не нужны. Подтверди, что поняла, и попроси продолжить план."
}, true);
assert.equal(acknowledgement.action,"reply");
assert.ok(acknowledgement.reply.trim());
assert.equal(acknowledgement.task,"");
console.log("PASS live contextual acknowledgement");

const task = "Сколько будет 2 + 2?";
const test = await interpret({plan:"Проверить арифметику", history:[],
    message:`Джессика, выполни один тест в своём чате. Точная задача: ${task}`});
assert.equal(test.action,"test");
assert.equal(test.task,task);
assert.equal(test.continueThread,false);
console.log("PASS live exact test extraction (test was not executed)");

const finish = await interpret({plan:"Все пункты выполнены", history:[],
    message:"План на сегодня полностью завершён. Тестов больше нет. Заверши сессию Кодекс."});
assert.equal(finish.action,"finish");
assert.ok(finish.reply.trim());
console.log("PASS live plan completion");
