import {verifyPairedSourceContent} from "../validator/pairedSourceContent.js";

const MIT = "https://opensource.org/license/mit";
const BSD = "https://opensource.org/license/bsd-3-clause";
const normalize = value => String(value || "").replace(/\s+/gu, " ").toLowerCase();

/** Deterministic table only after fresh complete operative license texts. */
export function groundedMitBsdComparison(task, plan, taskRunResult) {
    const request = String(task || "");
    if (!/(?:сравн|сопостав|compare)/iu.test(request) ||
        !/\bmit\b/iu.test(request) ||
        !/\bbsd[\s-]*3[\s-]*clause\b/iu.test(request) ||
        !/(?:таблиц|таблич|\btable\b)/iu.test(request) ||
        plan?.intent !== "native_known_license_comparison" ||
        verifyPairedSourceContent(plan, taskRunResult)?.valid !== true) return null;

    const results = (taskRunResult?.results || []).filter(r => r?.tool === "web_fetch");
    if (results.length !== 2) return null;
    const find = url => results.find(r => r?.success === true && r?.data?.url === url && r?.data?.truncated !== true);
    const mit = normalize(find(MIT)?.data?.content);
    const bsd = normalize(find(BSD)?.data?.content);
    const mitBegin = mit.indexOf("permission is hereby granted, free of charge");
    const mitEnd = mit.indexOf("other dealings in the software", mitBegin);
    const bsdBegin = bsd.indexOf("redistribution and use in source and binary forms");
    const bsdEnd = bsd.indexOf("even if advised of the possibility of such damage", bsdBegin);
    if (mitBegin < 0 || mitEnd < mitBegin + 600 || bsdBegin < 0 || bsdEnd < bsdBegin + 500) return null;
    const m = mit.slice(mitBegin, mitEnd + 30);
    const b = bsd.slice(bsdBegin, bsdEnd + 65);
    const mitClauses = [
        "permission is hereby granted, free of charge",
        "including without limitation the rights to use, copy, modify, merge, publish, distribute",
        "sublicense, and/or sell copies",
        "the above copyright notice and this permission notice shall be included",
        "the software is provided",
        "in no event shall the authors or copyright holders be liable",
        "other dealings in the software"
    ];
    const bsdClauses = [
        "redistribution and use in source and binary forms, with or without modification, are permitted",
        "1. redistributions of source code must retain",
        "2. redistributions in binary form must reproduce",
        "3. neither the name of the copyright holder",
        "without specific prior written permission",
        "this software is provided by the copyright holders and contributors",
        "even if advised of the possibility of such damage"
    ];
    if (!mitClauses.every(c => m.includes(c)) || !bsdClauses.every(c => b.includes(c))) return null;
    return {success: true, source: "bounded-license-comparison", text: [
        "| Критерий | MIT License | BSD 3-Clause |",
        "| --- | --- | --- |",
        "| Разрешённые действия | Разрешено использовать, копировать, изменять, объединять, публиковать, распространять, сублицензировать и продавать копии ПО. | Разрешены использование и распространение в исходной и бинарной формах, с изменениями или без них, при соблюдении условий лицензии. |",
        "| Уведомления при распространении | Уведомление об авторском праве и текст разрешения сохраняются во всех копиях или существенных частях ПО. | Для исходного кода сохраняются уведомление, условия и отказ от гарантий; для бинарной формы их воспроизводят в документации или сопроводительных материалах. |",
        "| Использование имён авторов | Отдельный пункт о продвижении с использованием имён не сформулирован в проверенном тексте. | Пункт 3 запрещает использовать имена правообладателя и участников для продвижения производных продуктов без письменного разрешения. |",
        "| Гарантии | ПО предоставляется «AS IS», без гарантий; ответственность авторов ограничена текстом лицензии. | ПО предоставляется «AS IS», без гарантий; ответственность правообладателей и участников ограничена текстом лицензии. |",
        "| Официальные тексты | " + MIT + " | " + BSD + " |"
    ].join("\n")};
}
