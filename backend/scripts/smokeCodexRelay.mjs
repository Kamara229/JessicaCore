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


const progress = await interpret({
    plan:"Продолжаем backend Jessica Core",history:[],
    message:[
        "JESSICA_REPLY_02bda0a8-6d96-47b9-96a2-5bdd91a2f2d0",
        "Тест пройден. Уточнил условия прекращения патентной лицензии Apache и добавил регрессию в CI. Коммит eab5003. Следующий шаг — проверить завершение CI и развёртывание на Render.",
        "JESSICA_END_02bda0a8-6d96-47b9-96a2-5bdd91a2f2d0"
    ].join("\n")
},true);
assert.equal(progress.action,"reply");
assert.equal(progress.task,"");
assert.match(progress.reply,/GitHub, Render и Supabase/);
console.log("PASS live status acknowledgement without model retries");


const simpleStep=await interpret({
 plan:"Jessica Core autonomous Learning",
 history:[],
 message:[
  "JESSICA_REPLY_e5fc766d-e1d9-4f47-97c3-c15cb91f3fc8",
  "Следующий шаг — проверить в Supabase, что опубликованные Skills имеют историю версий.",
  "JESSICA_END_e5fc766d-e1d9-4f47-97c3-c15cb91f3fc8"
 ].join("\n")
},true);
assert.equal(simpleStep.action,"reply");
assert.equal(simpleStep.task,"");
assert.match(simpleStep.reply,/Поняла следующий шаг/);
console.log("PASS live standalone next step without model retry");

const shortTest = "Сколько дней с 27 февраля по 2 марта 2024 года (27-е исключить, 2-е включить)? Объясни расчёт";
const compact = await interpret({plan:"Проверить обработку високосной даты",history:[],
    message:"Исправлен сдвиг даты. В обычном чате отправь: «"+shortTest+"». Пришли ответ дословно."
},true);
assert.equal(compact.action,"test");
assert.equal(compact.task,shortTest);
assert.equal(compact.continueThread,false);
console.log("PASS live compact test handoff without conversational model");

const failed = await interpret({
 plan:"Продолжить backend-проверку после неудачного теста",history:[],
 message:[
   "JESSICA_REPLY_766118ab-87f4-4052-abdc-0078eff08135",
   "Тест не пройден: селектор не выбрал первоисточник. Исправил backend: Apache 2.0 загружается напрямую с официального URL, без поиска (d09f42c). Следующий шаг — проверить CI и развёртывание Render; тест пока не повторять.",
   "JESSICA_END_766118ab-87f4-4052-abdc-0078eff08135"
 ].join("\n")
},true);
assert.equal(failed.action,"reply");
assert.equal(failed.task,"");
assert.match(failed.reply,/Тест в обычном чате Jessica пока не запрошен/);
console.log("PASS live failed-test next-step handoff");
