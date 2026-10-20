import test from "node:test";
import assert from "node:assert/strict";
import { buildComparisonEvidenceGrid,renderComparisonEvidenceGrid } from "../core/composer/comparisonEvidenceGrid.js";
import { buildComposerContext } from "../core/composer/composerContext.js";
import { parseClaimEvidenceResponse } from "../core/validator/claim/claimEvidenceResponse.js";
import { buildPartialComparisonEvidence } from "../core/composer/partialComparisonEvidence.js";

const apacheUrl="https://www.apache.org/licenses/LICENSE-2.0.html";
const bsdUrl="https://opensource.org/license/bsd-3-clause";
const apache=[
    "Apache License, Version 2.0",
    "Grant of Copyright License. Each Contributor grants to You a worldwide copyright license to reproduce, prepare Derivative Works of, sublicense and distribute the Work.",
    "Grant of Patent License. Each Contributor grants a perpetual worldwide non-exclusive, royalty-free patent license to use and distribute the Work.",
    "If You institute patent litigation against any entity alleging infringement, patent licenses granted under this License for that Work shall terminate.",
    "Redistribution. You must give any other recipients of the Work or Derivative Works a copy of this License and retain pertinent copyright notices.",
    "Trademarks. This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor."
].join("\n\n");
const bsd=[
    "BSD 3-Clause License",
    "Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:",
    "Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.",
    "Redistributions in binary form must reproduce the copyright notice, this list of conditions and the following disclaimer.",
    "Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products without prior written permission."
].join("\n\n");
const task="Сравни две лицензии Apache 2.0 и BSD 3-Clause: условия использования, распространения, сохранения уведомлений, патентные положения и ограничения.";
const plan={evidence:{mode:"source_content"},intent:"native_known_license_comparison"};
const run={results:[
    {tool:"web_fetch",success:true,data:{url:apacheUrl,title:"Apache 2.0",content:apache}},
    {tool:"web_fetch",success:true,data:{url:bsdUrl,title:"BSD 3-Clause",content:bsd}}
]};

test("source-first grid extracts criterion evidence from BOTH original documents",()=>{
    const grid=buildComparisonEvidenceGrid(task,run);
    assert.equal(grid.sources.length,2);
    assert.ok(grid.criteria.length>=5);
    const patent=grid.criteria.find(x=>x.id==="patents");
    assert.ok(patent);
    assert.ok(patent.excerpts[0].some(x=>x.includes("patent")));
    assert.equal(patent.excerpts[1].length,0);
    const rendered=renderComparisonEvidenceGrid(grid);
    assert.ok(rendered.includes(apacheUrl));
    assert.ok(rendered.includes(bsdUrl));
    assert.match(rendered,/Нет цитаты по критерию/);
    assert.doesNotMatch(rendered,/BSD(?: 3-Clause)? (?:не содержит|не имеет) патент/iu);
    const composer=buildComposerContext(task,plan,run);
    assert.match(composer,/КАРТА ПЕРВОИСТОЧНИКОВ/);
    assert.match(composer,/КРИТЕРИЙ: Патентные условия/);
    assert.ok(composer.includes(apacheUrl));
});

test("quote related to distribution does not prove notice-retention obligations",()=>{
    const raw=JSON.stringify({claims:[{
        claim:"Apache 2.0 требует сохранения уведомлений при распространении",
        supported:true,
        sourceUrl:"SOURCE 1",
        evidence:"sublicense and distribute the Work",
        reason:""
    }]});
    const source="SOURCE 1\nURL: "+apacheUrl+"\nSNIPPET: "+apache+
        "\n\nSOURCE 2\nURL: "+bsdUrl+"\nSNIPPET: "+bsd;
    const parsed=parseClaimEvidenceResponse(raw,source);
    assert.equal(parsed.success,true);
    assert.equal(parsed.claims[0].supported,false);
    assert.match(parsed.claims[0].reason,/Цитата не подтверждает (?:предмет|обязательное требование)/);
});

test("an explicit trademark non-grant can be supported without source-title mismatch",()=>{
    const quote="This License does not grant permission to use the trade names, trademarks";
    const raw=JSON.stringify({claims:[{
        claim:"Apache 2.0 не предоставляет прав на торговые марки",
        supported:true,
        sourceUrl:"SOURCE 1",
        evidence:quote,
        reason:""
    }]});
    const sources="SOURCE 1\nURL: "+apacheUrl+"\nSNIPPET: "+apache+
        "\n\nSOURCE 2\nURL: "+bsdUrl+"\nSNIPPET: "+bsd;
    const parsed=parseClaimEvidenceResponse(raw,sources);
    assert.equal(parsed.success,true);
    assert.equal(parsed.claims[0].supported,true);
});

test("partial comparison is arranged by criteria, not an unstructured dump",()=>{
    const claims=[
        {supported:true,sourceUrl:apacheUrl,evidence:"reproduce, prepare Derivative Works of, sublicense and distribute the Work"},
        {supported:true,sourceUrl:apacheUrl,evidence:"perpetual worldwide non-exclusive, royalty-free patent license"},
        {supported:true,sourceUrl:bsdUrl,evidence:"Redistribution and use in source and binary forms, with or without modification"},
        {supported:true,sourceUrl:bsdUrl,evidence:"Redistributions of source code must retain the above copyright notice"}
    ];
    const result=buildPartialComparisonEvidence(task,plan,run,claims);
    assert.ok(result);
    assert.match(result.text,/Частичный результат сравнения/);
    assert.match(result.text,/\| Критерий \|/);
    assert.match(result.text,/Патентные условия/);
    assert.match(result.text,/Не удалось подтвердить/);
    assert.ok(result.text.includes(apacheUrl)&&result.text.includes(bsdUrl));
    assert.equal(result.degraded,true);
});
