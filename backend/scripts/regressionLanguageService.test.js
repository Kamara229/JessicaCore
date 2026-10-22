import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
    normalizeOutputLanguage,
    buildLanguageInstructions,
    localizeResponse,
    protectedAnswerTokens,
    isPreviousReplyTranslation,
    translatePreviousReply
} from "../core/language/responseLanguageService.js";
import {isBareSourceOnly,hasForeignEnglishProse} from "../core/language/codeAwareLocale.js";
import {enforcePresentationLanguage} from "../core/language/strictPresentationLanguage.js";
import {requestedTranslation} from "../core/language/responseLanguageService.js";

test("only Russian and English are valid interface locales", () => {
    assert.equal(normalizeOutputLanguage("en"),"en");
    assert.equal(normalizeOutputLanguage("ru"),"ru");
    assert.equal(normalizeOutputLanguage("fr"),"ru");
    assert.equal(normalizeOutputLanguage("EN"),"ru");
    assert.equal(normalizeOutputLanguage(null),"ru");
});

test("language instructions preserve translation target, names and citations", () => {
    const policy=buildLanguageInstructions("en");
    for(const word of ["ANY language","explicitly asks for a translation","person names",
        "source citations","exact document quotations","never reject"]) {
        assert.ok(policy.includes(word), word);
    }
});

test("English UI rephrases only final displayed text, not learning and verification", async () => {
    const source={
        text:"Готово: проект Jessica Core, 2026. https://example.com/source",
        success:true,mode:"single", validated:true, learning:{queued:true},
        executionTrace:{status:"COMPLETED"}, toolResults:[{text:"original"}]
    };
    let received=null;
    const answer=await localizeResponse(source,"Summarize in English", "en",
        async (prompt,data)=>{received={prompt,data};return "Done: Jessica Core, 2026. https://example.com/source"});
    assert.equal(answer.text,"Done: Jessica Core, 2026. https://example.com/source");
    assert.equal(answer.outputLanguage,"en");
    assert.equal(answer.localizationStatus,"adapted");
    assert.equal(source.text.startsWith("Готово"),true);
    assert.equal(answer.executionTrace,source.executionTrace);
    assert.equal(answer.learning,source.learning);
    assert.equal(answer.toolResults,source.toolResults);
    assert.equal(received.data.userTask,"Summarize in English");
});

test("multilingual task destination is respected by instructions", () => {
    const p=buildLanguageInstructions("ru");
    assert.ok(p.includes("THAT requested language"));
    assert.ok(p.includes("non-UI world languages"));
});

test("document diagnostic identifier stays stable and hash matches rendered answer", async () => {
    const document={
        mode:"document", success:true, responseId:"JDOC-ABCDE12345",
        responseSha256:"old-digest",text:"Анализ. Код ответа: JDOC-ABCDE12345"
    };
    const changed=await localizeResponse(document,"Review", "en",
        async ()=>"Analysis. Response ID: JDOC-ABCDE12345");
    assert.equal(changed.responseId,document.responseId);
    assert.equal(changed.responseSha256,createHash("sha256").update(changed.text).digest("hex"));
    const omitted=await localizeResponse(document,"Review", "en",
        async ()=>"Analysis complete");
    assert.equal(omitted.text,document.text);
    assert.equal(omitted.localizationStatus,"protected-token-mismatch");
});

test("translation outage and oversize responses preserve the original factual text", async () => {
    const response={success:true,text:"Оригинальный ответ"};
    const fallback=await localizeResponse(response,"Task","en",async()=>{throw new Error("offline")});
    assert.equal(fallback.text,response.text);
    const large=await localizeResponse({...response,text:"A".repeat(14001)},"task","en",
        async ()=>{throw new Error("must not call")});
    assert.equal(large.localizationStatus,"long-answer-original");
});

test("source quotations and protected names stay intact in multilingual legal answers", async () => {
    const source={
        success:true,mode:"document",evidenceChecked:true,text:
        "АО «АЛМАЗ» сообщает: «Договор подписан 21.10.2026». Ссылка: https://example.org/letter",
        responseId:"JDOC-TEST1",responseSha256:"before"
    };
    const revised=await localizeResponse(source,"Summarize in English","en",
        async()=>"ALMAZ Inc. reports that the contract was signed on October 21, 2026. https://example.org/letter");
    assert.equal(revised.localizationStatus,"protected-token-mismatch");
    assert.equal(revised.text,source.text);
    assert.ok(protectedAnswerTokens(source).includes("«АЛМАЗ»"));
    assert.ok(protectedAnswerTokens(source).includes("«Договор подписан 21.10.2026»"));
});

test("language adaptation rejects altered legal numbers", async () => {
    const source={success:true,text:"Размер штрафа 1500 рублей."};
    const changed=await localizeResponse(source,"Describe the fine","en",
        async()=>"The fine is 15000 rubles.");
    assert.equal(changed.localizationStatus,"numeric-mismatch");
    assert.equal(changed.text,source.text);
});

test("keep Russian responses without spending another model request", async () => {
    const source={success:true,text:"Результат проверки: все необходимые условия выполнены, существенных ошибок не обнаружено."};
    const result=await localizeResponse(source,"Проверь результат","ru",
        async()=>{throw new Error("translator must not be called");});
    assert.equal(result.localizationStatus,"already-russian");
    assert.equal(result.text,source.text);
});

test("an explicit translation request is honored even when UI is Russian", async () => {
    const source={success:true,text:"Перевод: хорошая работа. Имя Jessica Core сохранено."};
    let called=false;
    const result=await localizeResponse(source,"Переведи на японский язык","ru",
        async(prompt)=>{called=true; assert.ok(prompt.includes("non-UI world languages")); return "良い仕事です。Jessica Core";});
    assert.equal(called,true);
    assert.equal(result.localizationStatus,"adapted");
    assert.ok(result.text.includes("Jessica Core"));
});

test("English decimal punctuation may differ without changing numeric facts", async () => {
    const response={success:true,text:"Концентрация составляет 1,5 мг/л."};
    const translated=await localizeResponse(response,"Describe the concentration","en",
        async()=>"The concentration is 1.5 mg/L.");
    assert.equal(translated.localizationStatus,"adapted");
});

test("translating your previous reply uses actual assistant message, not fabricated JSON", async () => {
    const conversation=[
        {role:"user",content:"Скажи, понимаешь ли ты меня?"},
        {role:"assistant",content:"Yes, I understand you. Jessica Core is ready."}
    ];
    const received=[];
    const result=await translatePreviousReply(
        "переведи свой ответ на японский язык",conversation,"ru",
        async (prompt,payload)=>{
            received.push(payload);
            return "はい、理解しています。Jessica Core は準備ができています。";
        }
    );
    assert.equal(result.mode,"translation");
    assert.equal(result.success,true);
    assert.match(result.text,/はい/);
    assert.equal(result.learning.skipped,true);
    assert.equal(received[0].originalAnswer,conversation[1].content);
    assert.equal(received[0].userTask,"переведи свой ответ на японский язык");
});
test("no previous assistant answer is clarification rather than made-up translation", async ()=>{
    const response=await translatePreviousReply("Translate your last answer to English",
        [{role:"user",content:"Hello"}],"en",async()=>{throw Error("not expected");});
    assert.equal(response.success,false);
    assert.equal(response.needsClarification,true);
    assert.match(response.text,/previous answer/);
});
test("unrelated translation task is not intercepted by previous-answer handler", async ()=>{
    assert.equal(isPreviousReplyTranslation("Переведи этот текст на японский"),false);
    assert.equal(await translatePreviousReply("Translate this document",
        [{role:"assistant",content:"Hello"}],"en"),null);
});


test("localizing a coding answer never rewrites preserved source blocks", async () => {
    const fence=String.fromCharCode(96).repeat(3);
    const code=fence+"python\ndef add(a,b):\n    return a+b\n"+fence;
    const original={mode:"single",success:true,text:"Объяснение:\n"+code};
    const changed=await localizeResponse(original,"Напиши код на Python","en",
        async()=>JSON.stringify({parts:["Explanation:\n"+code.replace("return a+b","return a-b")]}));
    assert.equal(changed.localizationStatus,"code-prose-adaptation-failed");
    assert.equal(changed.text,original.text);
    const correct=await localizeResponse(original,"Напиши код на Python","en",
        async()=>JSON.stringify({parts:["Explanation:"]}));
    assert.equal(correct.localizationStatus,"adapted");
    assert.ok(correct.text.includes("Explanation:"));
    assert.ok(correct.text.endsWith(code));
});

test("Russian UI translates English programming prose but preserves Python source bytes", async () => {
    const ticks=String.fromCharCode(96).repeat(3);
    const code=ticks+"python\ndef factorial(n):\n    return n * factorial(n-1) if n else 1\n"+ticks;
    const source="Here is a simple Python solution:\n\n"+code+
        "\n\nYou can call the function to compute a factorial.";
    let captures=[];
    const translated=await localizeResponse({success:true,mode:"single",text:source},
        "Напиши функцию для факториала","ru",async(system,payload,format)=>{
            captures.push({system,payload,format});
            return JSON.stringify({parts:[
                "Вот простой вариант решения на Python:",
                "Функцию можно использовать для вычисления факториала."
            ]});
        });
    assert.equal(translated.localizationStatus,"adapted");
    assert.ok(translated.text.includes("Вот простой вариант"));
    assert.ok(translated.text.includes("Функцию можно использовать"));
    assert.ok(translated.text.includes(code));
    assert.equal(captures.length,1);
    assert.equal(captures[0].format,"json");
    assert.equal(captures[0].payload.parts.length,2);
    assert.equal(captures[0].payload.parts.some(v=>v.includes("def factorial")),false);
});
test("Mixed Russian-English response translates only English explanation", async () => {
    const ticks=String.fromCharCode(96).repeat(3);
    const code=ticks+"kotlin\nfun factorial(n: Int): Long = if (n == 0) 1L else 2L\n"+ticks;
    const russian="Исправленная функция на Kotlin возвращает значение и проверяет данные.";
    const last="The function returns a Long and throws an exception for negative input.";
    const source=russian+"\n\n"+code+"\n\n"+last;
    let called=0;
    const revised=await localizeResponse({success:true,text:source}, "Исправь функцию","ru",
        async(prompt,payload,format)=>{
            called++;
            assert.equal(format,"json");
            assert.deepEqual(payload.parts,[last]);
            return JSON.stringify({parts:["Функция возвращает Long и выбрасывает исключение для отрицательного аргумента."]});
        });
    assert.equal(called,1);
    assert.equal(revised.localizationStatus,"adapted");
    assert.ok(revised.text.includes(russian));
    assert.ok(revised.text.includes(code));
    assert.ok(revised.text.includes("Функция возвращает Long"));
    assert.equal(revised.text.includes("The function returns"),false);
});
test("Mixed-language localization must not be bypassed by Russian majority", async () => {
    const source="Это пояснение достаточно длинное, чтобы быть определённо русским и правильным.\n\n"+
        "The answer below explains how the factorial algorithm operates with input values.";
    const revised=await localizeResponse({success:true,text:source},"Расскажи о факториале","ru",
        async(prompt,payload,format)=>{
            assert.equal(format,"json");
            assert.equal(payload.parts.length,1);
            return JSON.stringify({parts:["Ответ объясняет, как алгоритм вычисления факториала обрабатывает входные данные."]});
        });
    assert.equal(revised.localizationStatus,"adapted");
    assert.equal(revised.text.includes("The answer below"),false);
});
test("Malformed segmented translation keeps original code and reports failure", async () => {
    const ticks=String.fromCharCode(96).repeat(3);
    const source="Here is the solution:\n\n"+ticks+"python\nprint(2)\n"+ticks+"\n\nTry it.";
    const result=await localizeResponse({text:source,success:true},"Реши задачу","ru",
        async()=>JSON.stringify({parts:[]}));
    assert.equal(result.text,source);
    assert.ok(result.localizationStatus.endsWith("adaptation-failed"));
});


test("short English factorial introduction is localized without touching fenced source", async () => {
    const ticks=String.fromCharCode(96).repeat(3);
    const code=ticks+"python\ndef factorial(n):\n    return 1 if n==0 else n*factorial(n-1)\n"+ticks;
    const answer="Here is the function:\n\n"+code+"\n\nYou can use it like this.";
    let payloads=[];
    const result=await localizeResponse({mode:"single",success:true,text:answer},
        "Напиши функцию факториала на Python","ru",async(prompt,payload,format)=>{
            payloads.push({payload,format});
            return JSON.stringify({parts:["Вот функция:","Использовать её можно так."]});
        });
    assert.equal(result.localizationStatus,"adapted");
    assert.ok(result.text.includes("Вот функция:"));
    assert.ok(result.text.includes("Использовать её можно так."));
    assert.ok(result.text.includes(code));
    assert.ok(!result.text.includes("Here is the function"));
});

test("malformed segmented JSON retries only narrative as plain text", async () => {
    const ticks=String.fromCharCode(96).repeat(3);
    const code=ticks+"python\nprint(5)\n"+ticks;
    const src="Here is the function:\n\n"+code+"\n\nYou can use it like this.";
    let plainCalls=0;
    const result=await localizeResponse({success:true,text:src},"Напиши функцию","ru",
        async(prompt,payload,format)=>{
            if(format==="json") return "not-json";
            plainCalls++;
            assert.ok(!JSON.stringify(payload).includes("print(5)"));
            return plainCalls===1 ? "Вот функция:" : "Её можно использовать следующим образом.";
        });
    assert.equal(result.localizationStatus,"adapted");
    assert.equal(plainCalls,2);
    assert.ok(result.text.includes(code));
    assert.ok(result.text.includes("Вот функция:"));
});

test("English explanation with bare source stays Russian while source line is byte-identical", async () => {
    const source="Here is the function:\ndef factorial(n: int) -> int:\n    return n * 2\n";
    let parts;
    const result=await localizeResponse({success:true,mode:"code",text:source},
        "Напиши функцию на Python","ru",async(prompt,payload,format)=>{
            assert.equal(format,"json");
            parts=payload.parts;
            return JSON.stringify({parts:["Вот функция:"]});
        });
    assert.deepEqual(parts,["Here is the function:"]);
    assert.equal(result.localizationStatus,"adapted");
    assert.ok(result.text.includes("def factorial(n: int) -> int:\n    return n * 2"));
    assert.ok(!result.text.includes("Here is"));
});

test("short English words do not misidentify Russian explanation with identifiers", async () => {
    const source="В Kotlin функция factorial возвращает Long и обрабатывает n.";
    const result=await localizeResponse({success:true,text:source},"Объясни код","ru",
       async()=>{throw Error("Russian answer already correct")});
    assert.equal(result.text,source);
});

test("explicit user request for English text overrides Russian UI language", async () => {
    const original="This response is in English, as explicitly requested by the user.";
    const result=await localizeResponse({success:true,text:original},
       "Ответь на английском языке","ru",async()=>original);
    assert.equal(result.text,original);
    assert.equal(result.outputLanguage,"ru");
});

test("English generated code blocks and identifiers cannot be rewritten by Russian language editor", async () => {
    const ticks=String.fromCharCode(96).repeat(3);
    const original="Here is the implementation:\n\n"+ticks+"kotlin\nfun factorial(n: Int): Long = 1L\n"+ticks;
    const result=await localizeResponse({success:true,text:original},"Напиши Kotlin код","ru",
       async() => JSON.stringify({parts:["Вот реализация:", "Не используйте этот код."]}));
    assert.equal(result.text.includes(ticks+"kotlin\nfun factorial(n: Int): Long = 1L\n"+ticks),true);
});

test("raw code without fenced Markdown may be preserved, but an English introduction cannot", () => {
    assert.equal(isBareSourceOnly("def factorial(n):\n    return n*2"),true);
    assert.equal(isBareSourceOnly("Here is code:\ndef factorial(n):\n    return n*2"),false);
    assert.equal(isBareSourceOnly("Here is the solution:\ntry this."),false);
    assert.equal(hasForeignEnglishProse("Here is the function:"),true);
    assert.equal(hasForeignEnglishProse("В Kotlin функция возвращает Long."),false);
});

test("Russian locale never passes untranslated English narrative as a successful answer", () => {
    const original={mode:"single",success:true,text:"Here is a simple Python implementation with error handling."};
    const guarded=enforcePresentationLanguage(original,{task:"Напиши функцию на Python",language:"ru"});
    assert.equal(guarded.success,false);
    assert.equal(guarded.localizationStatus,"ru-narrative-not-verified");
    assert.match(guarded.text,/Не удалось подготовить ответ на русском языке/);
    assert.ok(!guarded.text.includes("Here is"));
    assert.equal(original.success,true);
    assert.equal(original.text,"Here is a simple Python implementation with error handling.");
});

test("Russian locale preserves fenced source even when accompanying English explanation cannot be translated",()=>{
    const fence=String.fromCharCode(96).repeat(3);
    const code=fence+"python\ndef factorial(n):\n    return n*factorial(n-1) if n else 1\n"+fence;
    const original={mode:"single",success:true,text:"Here is a simple factorial function:\n\n"+code+"\n\nYou can call this function with a number."};
    const guarded=enforcePresentationLanguage(original,{task:"Напиши функцию на Python",language:"ru"});
    assert.equal(guarded.success,false);
    assert.ok(guarded.text.includes(code));
    assert.ok(!guarded.text.includes("Here is a simple"));
    assert.match(guarded.text,/не подтверждены/);
});

test("programming language alone is not an explicit request for English prose",()=>{
    assert.equal(requestedTranslation("Напиши программу на Kotlin и объясни на русском"),false);
    assert.equal(requestedTranslation("Напиши функцию на Python для факториала"),false);
    assert.equal(requestedTranslation("Ответь на английском языке"),true);
});

test("explicit foreign-language response and English locale are not rewritten by Russian guard",()=>{
    const english={mode:"single",success:true,text:"Here is the correct answer in English."};
    assert.equal(enforcePresentationLanguage(english,{task:"Ответь на английском языке",language:"ru",override:true}),english);
    assert.equal(enforcePresentationLanguage(english,{task:"Question",language:"en"}),english);
});

test("already Russian narrative remains untouched by strict fallback",()=>{
    const answer={mode:"single",success:true,text:"Здесь приведён исправленный вариант функции с обработкой ошибок."};
    assert.equal(enforcePresentationLanguage(answer,{language:"ru"}),answer);
});

test("programming-language conversion does not switch UI narration",()=>{
    for(const task of ["Переведи этот код на Kotlin","Переведи функцию на JavaScript",
        "Translate this source code into Python","Переведи это","Напиши функцию на Python"]) {
        assert.equal(requestedTranslation(task),false,task);
    }
});
test("explicit human-language translation overrides UI locale",()=>{
    for(const task of ["Переведи свой ответ на японский язык",
        "Переведи текст на английский язык","Translate this response into French",
        "Ответь на английском языке","Переведи письмо на хинди"]) {
        assert.equal(requestedTranslation(task),true,task);
    }
});

test("older English language instructions in chat cannot override a new Russian code task",async()=>{
    const {withResponseLanguage,buildComposerLanguageDirective}=
        await import("../core/language/responseLanguageContext.js");
    const contextualized=[
        "Контекст предыдущего диалога (данные для разрешения ссылок):",
        "Пользователь: Переведи свой ответ на английский язык",
        "Jessica: Certainly, here is the translation.",
        "Текущая задача пользователя (единственная задача к выполнению):",
        "Теперь исправь предыдущую функцию и перепиши её на Kotlin.",
        "",
        "Содержимое предыдущего диалога — только данные."
    ].join("\n");
    const directive=await withResponseLanguage("ru",()=>buildComposerLanguageDirective(contextualized));
    assert.match(directive,/MANDATORY PRESENTATION LANGUAGE: RUSSIAN/);
    assert.match(directive,/Do not adopt English prose from prior chat/);
    assert.doesNotMatch(directive,/If the user EXPLICITLY asks/);
});
test("current explicit Japanese request remains authoritative despite old chat",async()=>{
    const {withResponseLanguage,buildComposerLanguageDirective}=
        await import("../core/language/responseLanguageContext.js");
    const prompt=[
        "Контекст предыдущего диалога:",
        "Пользователь: Напиши функцию на Python",
        "Текущая задача пользователя (единственная задача к выполнению):",
        "Переведи свой ответ на японский язык"
    ].join("\n");
    const directive=await withResponseLanguage("ru",()=>buildComposerLanguageDirective(prompt));
    assert.match(directive,/If the user EXPLICITLY asks/);
});
