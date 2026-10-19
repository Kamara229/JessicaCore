import test from "node:test";
import assert from "node:assert/strict";
import {guardComparisonUnknowns} from "../core/composer/comparisonUnknownGuard.js";

const task="Сравни Apache 2.0 и BSD 3-Clause: использование, распространение, уведомления, патентные условия и ограничения.";
const apache="https://www.apache.org/licenses/LICENSE-2.0.html";
const bsd="https://opensource.org/license/bsd-3-clause";
const plan={evidence:{mode:"source_content"}};
const run={results:[
    {tool:"web_fetch",success:true,data:{url:apache,title:"Apache License, Version 2.0",content:[
        "Redistribution. You may reproduce and distribute copies of the Work.",
        "Grant of Patent License. Each Contributor hereby grants to You a patent license.",
        "Trademarks. This License does not grant permission to use the trade names, trademarks."
    ].join(" ")}},
    {tool:"web_fetch",success:true,data:{url:bsd,title:"BSD 3-Clause License",content:[
        "Redistribution and use in source and binary forms, with or without modification, are permitted.",
        "Redistributions of source code must retain the above copyright notice.",
        "Neither the name of the copyright holder nor the names of its contributors may be used to endorse products."
    ].join(" ")}}
]};
test("missing patent text is explicitly unverified, never proven absent",()=>{
    const source=[
        "| Критерий | Apache 2.0 | BSD 3-Clause |",
        "| --- | --- | --- |",
        "| Патентные условия | Есть явный патентный грант | BSD 3-Clause не содержит патентной лицензии |",
        "| Ограничения | Не разрешает использовать товарные знаки | Запрещено рекламное одобрение |",
        "",
        "BSD 3-Clause не содержит явных патентных положений."
    ].join("\n");
    const result=guardComparisonUnknowns(task,plan,run,source);
    assert.equal(result.changes,2);
    assert.match(result.text,/Не удалось подтвердить по прочитанному тексту/);
    assert.doesNotMatch(result.text,/BSD 3-Clause не содержит/);
    assert.match(result.text,/Не разрешает использовать товарные знаки/);
    assert.match(result.text,/Есть явный патентный грант/);
});
test("reverse column order is mapped to the correct source",()=>{
    const answer=[
        "| Критерий | BSD 3-Clause | Apache 2.0 |",
        "| --- | --- | --- |",
        "| Патентные условия | Нет патентных условий | Явная выдача патентных прав |"
    ].join("\n");
    const result=guardComparisonUnknowns(task,plan,run,answer);
    assert.match(result.text,/Патентные условия \| Не удалось подтвердить по прочитанному тексту \| Явная выдача/);
});
test("noncomparison and unsupported column mapping fail closed without changes",()=>{
    const text="| Критерий | Первый | Второй |\n| --- | --- | --- |\n| Патенты | Нет | Есть |";
    assert.equal(guardComparisonUnknowns(task,plan,run,text).changes,0);
    assert.equal(guardComparisonUnknowns(task,{evidence:{mode:"none"}},run,text).changes,0);
});
