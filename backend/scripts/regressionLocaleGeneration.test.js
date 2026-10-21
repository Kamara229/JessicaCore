import test from "node:test";
import assert from "node:assert/strict";
import {
    withResponseLanguage,
    currentResponseLanguage,
    buildComposerLanguageDirective
} from "../core/language/responseLanguageContext.js";
import {getComposerInstructions} from "../core/composer/composerPrompt.js";

test("Russian is the safe default for regular programming answers", () => {
    assert.equal(currentResponseLanguage(),"ru");
    const instruction = getComposerInstructions("Напиши функцию факториала на Python");
    assert.match(instruction,/MANDATORY PRESENTATION LANGUAGE: RUSSIAN/);
    assert.match(instruction,/programming language.*NOT.*surrounding explanatory prose/i);
    assert.match(instruction,/Preserve programming syntax, code blocks/);
});

test("selected English overrides Russian task language, but not code keywords", async() => {
    await withResponseLanguage("en", async () => {
        await Promise.resolve();
        assert.equal(currentResponseLanguage(),"en");
        assert.match(getComposerInstructions("Напиши функцию на Kotlin"),/LANGUAGE: ENGLISH/);
    });
    assert.equal(currentResponseLanguage(),"ru");
});

test("request language is isolated between overlapping async requests", async () => {
    let done;
    const pending = new Promise(resolve => {done=resolve;});
    const russian = withResponseLanguage("ru", async () => {
        await pending;
        return currentResponseLanguage();
    });
    const english = withResponseLanguage("en", async () => {
        await Promise.resolve();
        done();
        return currentResponseLanguage();
    });
    assert.deepEqual(await Promise.all([russian,english]),["ru","en"]);
});

test("translation instructions retain explicitly requested target language", () => {
    const directive=buildComposerLanguageDirective("Переведи текст на японский язык");
    assert.match(directive,/EXPLICITLY asks to translate/);
});
