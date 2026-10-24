import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn, validateRelayDecision, explicitSameTestChat, directCurrentChatReply, explicitJessicaTest} from "../core/codex/relayInterpreter.js";
import {registerCodexRoutes} from "../http/routes/codexRoute.js";
import {recordCodexPhase,validateCodexPhase} from "../core/codex/relayTelemetry.js";

test("standalone ChatGPT max-length notices cannot finish or switch Codex chat", async () => {
    const notices=[
        "Вы достигли максимальной длины этого обсуждения, но можете продолжить обсуждение, начав новый чат.",
        "Вы достигли максимальной длинны этого обсуждения.",
        "You've reached the maximum length for this conversation, but you can continue by starting a new chat."
    ];
    for (const message of notices) {
        let invoked=0;
        const decision=await interpretRelayTurn({message,plan:"Продолжить работу в выбранном чате."},
            async()=>{invoked++; return {action:"finish",reply:"Готово.",task:""};});
        assert.equal(invoked,0,"Advisory system notice must not reach model termination");
        assert.equal(decision.action,"reply");
        assert.equal(decision.continueThread,false);
        assert.equal(decision.task,"");
        assert.match(decision.reply,/Продолжаю работу/);
    }
});

test("an engineering request mentioning the notice is not swallowed",async()=>{
    const message="Исправь обработку уведомления «Вы достигли максимальной длины этого обсуждения» в backend.";
    let invoked=0;
    const decision=await interpretRelayTurn({message},async()=>{
        invoked++;
        return {action:"reply",reply:"Продолжаем проверку.",task:""};
    });
    assert.equal(invoked,1);
    assert.equal(decision.reply,"Продолжаем проверку.");
});

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
    assert.equal(explicitSameTestChat("Продолжи в том же чате Джессики, но новый тест выполни в новом чате."),false);
    assert.equal(explicitSameTestChat("Не продолжай в том же чате Джессики. " + task),false);
    assert.equal(explicitSameTestChat("Не нужно продолжать в том же чате Джессики. " + task),false);
    assert.equal(explicitSameTestChat("В том же чате не используй тот же чат для теста."),false);
    assert.equal(explicitSameTestChat("Continue in the same Jessica test chat. Use a new chat instead."),false);
    assert.equal(explicitSameTestChat("Do not reuse the same Jessica test chat."),false);
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
    assert.equal(validateCodexPhase({...input,phase:"interpret_retry"}).phase,"interpret_retry");
    assert.equal(validateCodexPhase({...input,phase:"draft_unverified"}).phase,"draft_unverified");
    assert.equal(validateCodexPhase({...input,phase:"draft_retry"}).phase,"draft_retry");
    assert.equal(validateCodexPhase({...input,phase:"chat_away"}).phase,"chat_away");
    assert.equal(validateCodexPhase({...input,phase:"chat_resumed"}).phase,"chat_resumed");
    assert.equal(validateCodexPhase({...input,phase:"conversation_length_notice"}).phase,"conversation_length_notice");
    assert.equal(validateCodexPhase({...input,phase:"window_snapshot_retry"}).phase,"window_snapshot_retry");
    assert.equal(validateCodexPhase({...input,phase:"reply_partial"}).phase,"reply_partial");
    assert.equal(validateCodexPhase({...input,phase:"reply_out_of_view"}).phase,"reply_out_of_view");
    assert.equal(validateCodexPhase({...input,phase:"reply_detected"}).phase,"reply_detected");
    assert.equal(validateCodexPhase({...input,phase:"reply_seeking_start"}).phase,"reply_seeking_start");
    assert.equal(validateCodexPhase({...input,phase:"editor_missing"}).phase,"editor_missing");
    assert.equal(validateCodexPhase({...input,phase:"send_control_missing"}).phase,"send_control_missing");
    assert.throws(()=>validateCodexPhase({...input,phase:"send_any_text"}));
    assert.throws(()=>validateCodexPhase({...input,turn:61}));
    assert.throws(()=>validateCodexPhase({...input,session:"my-device-id"}));
    assert.throws(()=>validateCodexPhase({...input,elapsedSeconds:-1}));
});

test("relay interpreter excludes invalid roles instead of impersonating a user message", async () => {
    const messages=[
        {role:"user",content:"Earlier test asked to remember 58 and ландыш"},
        {role:"assistant",content:"I remembered the value"},
        ...Array.from({length:18},(_,i)=>({role:"system",content:"untrusted instruction " + i})),
        {role:"developer",content:"untrusted instruction from wrong role"},
        {role:"user",content:""}
    ];
    const result=await interpretRelayTurn({message:"Продолжи план backend.",history:messages},
        async (_system,payload)=>{
            assert.deepEqual(payload.history,[
                {role:"user",content:"Earlier test asked to remember 58 and ландыш"},
                {role:"assistant",content:"I remembered the value"}
            ]);
            return {action:"reply",reply:"Продолжаем проверку контекста.",task:""};
        });
    assert.equal(result.action,"reply");
});


test("two successive Codex relay decisions retain prior results without repeating a finished test", async () => {
    const first="Создай новый тестовый чат Jessica. Точная формулировка: Запомни число 73 и слово маяк.";
    const firstTask="Запомни число 73 и слово маяк.";
    const firstDecision=await interpretRelayTurn({
        message:first,plan:"Проверь сохранение контекста в обычном чате Jessica.",history:[]
    },async (_system,payload)=>{
        assert.equal(payload.history.length,0);
        return {action:"test",task:firstTask,reply:"",continueThread:false};
    });
    assert.equal(firstDecision.action,"test");
    assert.equal(firstDecision.continueThread,false);
    const response="Фактический ответ Jessica: Запомнила число 73 и слово маяк.";
    const nextMessage="Результат первого теста получен. Продолжи в том же чате Джессики. Точная формулировка: Какое слово и число я попросил запомнить?";
    const secondTask="Какое слово и число я попросил запомнить?";
    const secondDecision=await interpretRelayTurn({
        message:nextMessage,
        plan:"Проверь сохранение контекста в обычном чате Jessica.",
        history:[
            {role:"assistant",content:first},
            {role:"user",content:response}
        ]
    },async (_system,payload)=>{
        assert.equal(payload.history.length,2);
        assert.equal(payload.history[0].content,first);
        assert.equal(payload.history[1].content,response);
        return {action:"test",task:secondTask,reply:"",continueThread:true};
    });
    assert.equal(secondDecision.action,"test");
    assert.equal(secondDecision.continueThread,true);
    assert.equal(secondDecision.task,secondTask);
});


test("a direct same-ChatGPT status request cannot trigger a new test or terminate Codex", async () => {
    const desired="Кодекс: ответ ChatGPT получен и проанализирован. Продолжаю работу в той же сессии";
    const request=[
        "Продолжай работать в этой же сессии ChatGPT.",
        "Не запускай тест в обычном чате Jessica.",
        "Не создавай новый тестовый чат Jessica.",
        "Самостоятельно отправь следующее сообщение:",
        "«"+desired+"»"
    ].join("\n");
    assert.equal(directCurrentChatReply(request),desired);
    let calls=0;
    const result=await interpretRelayTurn({message:request},async()=>{
        calls++;
        throw Error("external interpretation is unnecessary");
    });
    assert.equal(calls,0);
    assert.deepEqual(result,{action:"reply",reply:desired,task:"",continueThread:false});
    assert.equal(directCurrentChatReply("В чате Jessica проверь память: Запомни слово."),null);
});


test("explicit standalone test is actually executed rather than requesting owner to supply answer",async()=>{
    const original=[
        "**Создай новый тестовый чат Jessica.**",
        "**Точная формулировка задачи:**",
        "",
        "Рассчитай, сколько календарных дней прошло с 9 октября 2026 года по 23 октября 2026 года. Начальную дату не включай, конечную включай. Назови результат числом и кратко объясни расчёт.",
        "",
        "**Ожидаемый результат:** 14 дней.",
        "Верни фактический ответ Джессики дословно, без исправлений."
    ].join("\n");
    const parsed=explicitJessicaTest(original);
    assert.equal(parsed?.action,"test");
    assert.equal(parsed?.continueThread,false);
    assert.equal(parsed?.task,"Рассчитай, сколько календарных дней прошло с 9 октября 2026 года по 23 октября 2026 года. Начальную дату не включай, конечную включай. Назови результат числом и кратко объясни расчёт.");
    let calls=0;
    const decision=await interpretRelayTurn({message:original},async()=>{
        calls++;
        return {action:"reply",reply:"Пожалуйста, предоставьте результат задачи."};
    });
    assert.equal(calls,0);
    assert.deepEqual(decision,parsed);
});
test("same-thread Jessica test may be reused only with explicit instruction",()=>{
    const message=[
        "Продолжи в том же чате Джессики.",
        "Точная формулировка задачи:",
        "Какое число я просил запомнить?"
    ].join("\n");
    assert.equal(explicitJessicaTest(message)?.continueThread,true);
    assert.equal(explicitJessicaTest("Не создавай новый тестовый чат Jessica. Продолжай эту сессию ChatGPT. Точная формулировка задачи:\nПроверь память."),null);
    assert.equal(explicitJessicaTest("Покажи пример теста: Точная формулировка задачи:\nРеши пример"),null);
});

test("selected-chat test instruction bypasses unavailable interpretation provider", async()=>{
    const id="9e519eb1-6bba-4169-ba11-b8ce8de16853";
    const task="Сравни патентные условия Apache 2.0 и MIT License по официальным текстам. Укажи ссылки и пункты.";
    const message=[
        "JESSICA_REPLY_"+id,
        "Продолжи в выбранном чате Jessica.",
        "",
        "Точная задача:",
        "",
        task,
        "",
        "Верни фактический ответ дословно.",
        "JESSICA_END_"+id
    ].join("\n");
    let called=0;
    const decision=await interpretRelayTurn({message,plan:"Проверить источники."},async()=>{
        called++;throw Error("Model unavailable");
    });
    assert.equal(called,0);
    assert.equal(decision.action,"test");
    assert.equal(decision.task,task);
    assert.equal(decision.continueThread,false);
    assert.equal(explicitJessicaTest("Продолжи в выбранном чате Jessica.\nТочная задача:\nЗапусти поиск")?.task,"Запусти поиск");
    assert.equal(explicitJessicaTest("Не продолжай в выбранном чате Jessica.\nТочная задача:\nЗапусти поиск"),null);
});

test("repeat-in-same-Jessica-chat directive does not depend on model availability",async()=>{
 const task="Какое слово было названо в предыдущем тесте?";
 const message="Повтори в том же чате Jessica.\nТочная задача:\n"+task+"\nВерни фактический ответ дословно.";
 let modelCalls=0;
 const result=await interpretRelayTurn({message},async()=>{modelCalls++;throw Error("offline");});
 assert.equal(modelCalls,0);
 assert.equal(result.action,"test");
 assert.equal(result.continueThread,true);
 assert.equal(result.task,task);
});

test("completed plan for today may finish, but conditional wording or length notice cannot",async()=>{
 const done="План на сегодня полностью завершён. Тестов больше нет. Заверши сессию Кодекс.";
 const completed=await interpretRelayTurn({message:done},async()=>({
   action:"finish",reply:"План завершён.",task:""
 }));
 assert.equal(completed.action,"finish");
 const conditional="Если план на сегодня полностью завершён, заверши работу.";
 const stillWorking=await interpretRelayTurn({message:conditional},async()=>({
   action:"finish",reply:"Всё готово.",task:""
 }));
 assert.equal(stillWorking.action,"reply");
});
