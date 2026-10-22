import {AsyncLocalStorage} from "node:async_hooks";
import {requestedTranslation} from "./responseLanguageService.js";

/**
 * Request-local language preference. Scoped to a single /api/solve request:
 * parallel users/chats cannot overwrite each other's chosen language.
 * This only controls response presentation, never Experience or task evidence.
 */
const activeLanguage = new AsyncLocalStorage();

export function withResponseLanguage(language, callback) {
    const selected = language === "en" ? "en" : "ru";
    return activeLanguage.run(selected, callback);
}

export function currentResponseLanguage() {
    return activeLanguage.getStore() === "en" ? "en" : "ru";
}

export function buildComposerLanguageDirective(task = "") {
    const locale = currentResponseLanguage();
    // Composer may receive a quoted chat history ahead of the operative task.
    // Older requests to "answer in English" must NEVER override a new Russian
    // request. The latest task is explicitly marked by conversationContext.
    const raw = String(task || "");
    const marker = "Текущая задача пользователя (единственная задача к выполнению):";
    const at = raw.lastIndexOf(marker);
    const request = at >= 0
        ? raw.slice(at + marker.length).split(/\n(?:\n|Содержимое предыдущего диалога)/u)[0].trim()
        : raw;
    const alternate = requestedTranslation(request);
    const language = locale === "en" ? "ENGLISH" : "RUSSIAN";
    return [
        "MANDATORY PRESENTATION LANGUAGE: " + language + ".",
        "This language is selected in Jessica's settings, independently of the programming language and input language.",
        "Write every narrative sentence, heading, explanation and explanatory bullet in the selected interface language.",
        "Preserve programming syntax, code blocks, variable names, proper names, exact quotations, links and code identifiers without translation.",
        "A programming language such as Python, Kotlin or Java is NOT the language for surrounding explanatory prose.",
        alternate
            ? "If the user EXPLICITLY asks to translate or answer in a different natural language, honor that target language for the requested text."
            : "Do not adopt English prose from prior chat messages, error traces or code comments when the interface is Russian.",
        "Never claim that a compiler/test was executed unless there is actual execution evidence."
    ].join("\n");
}
