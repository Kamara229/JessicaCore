/*
 * =========================================================
 * JESSICA RELEASE FACT TOOL FACADE v3
 * =========================================================
 *
 * Тонкий facade:
 *
 * Tool Registry
 *      ↓
 * releaseFactExecutor
 *      ↓
 * text / section / dates / versions / candidates / evidence
 *
 * Вся бизнес-логика разнесена по маленьким модулям.
 * =========================================================
 */


import {
    registerTool
} from "./toolRegistry.js";


import {
    executeReleaseFact
} from "./releaseFact/releaseFactExecutor.js";


registerTool({

    name:
        "release_fact",

    description:

        (
            "Детерминированно извлекает актуальную стабильную версию "
            +
            "и дату выпуска запрошенного продукта из уже загруженного "
            +
            "официального источника. Не использует AI."
        ),

    arguments: {

        content:
            "Текст уже загруженного официального источника.",

        sourceUrl:
            "URL источника для provenance.",

        sourceTitle:
            "Заголовок загруженной страницы.",

        subject:
            "Исходная формулировка искомого продукта или релиза."

    },

    execute:
        executeReleaseFact

});
