import "../tools/webFetchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {
    buildKnownLicenseComparisonPlan,
    tryCreateKnownLicenseComparison
} from "../core/planner/knownLicenseComparison.js";

test("Apache-2.0 vs BSD-3-Clause reads canonical legal texts, not project PDF", () => {
    const task = "Сравни две лицензии программного обеспечения: Apache License 2.0 и BSD 3-Clause. Найди официальные тексты обеих лицензий и изучи их содержание.";
    const plan = buildKnownLicenseComparisonPlan(task);
    assert.ok(plan);
    assert.equal(plan.steps.length, 2);
    assert.deepEqual(plan.steps.map(x => x.tool), ["web_fetch","web_fetch"]);
    assert.deepEqual(new Set(plan.steps.map(x => x.arguments.url)), new Set([
        "https://www.apache.org/licenses/LICENSE-2.0.html",
        "https://opensource.org/license/bsd-3-clause"
    ]));
    assert.equal(plan.evidence.mode,"source_content");
    assert.equal(tryCreateKnownLicenseComparison({task}).handled,true);
});
test("MIT and BSD 2-Clause can also be located from canonical URL registry", () => {
    const p=buildKnownLicenseComparisonPlan("Сравни MIT License и BSD 2-Clause: изучи официальные тексты лицензий");
    assert.ok(p);
    assert.equal(p.steps.length,2);
});
test("do not replace user URLs or invent mappings for unknown licenses", () => {
    assert.equal(buildKnownLicenseComparisonPlan("Сравни Apache License 2.0 и BSD 3-Clause: https://first.example https://second.example"),null);
    assert.equal(buildKnownLicenseComparisonPlan("Расскажи про Apache License 2.0"),null);
    assert.equal(buildKnownLicenseComparisonPlan("Сравни Apache License 2.0, BSD 3-Clause и MIT License"),null);
    assert.equal(buildKnownLicenseComparisonPlan("Сравни две неизвестные лицензии"),null);
});

test("verification covers known comparison cells and honestly unverified clauses", async () => {
    const { checkPairedClaimCoverage } = await import("../core/validator/claim/pairedClaimCoverage.js");
    const task = "Сравни лицензии: права, распространение, уведомления, патентные условия и ограничения";
    const urls = ["https://www.apache.org/licenses/LICENSE-2.0.html",
        "https://opensource.org/license/bsd-3-clause"];
    const run = {results:urls.map(url => ({tool:"web_fetch",success:true,data:{url}}))};
    const plan = {evidence:{mode:"source_content"}};
    const answer=[
        "| Критерий | Apache | BSD |",
        "|---|---|---|",
        "| Использование | Да | Да |",
        "| Распространение | Да | Да |",
        "| Уведомления | Да | Да |",
        "| Патентные условия | Есть явный пункт | Не удалось подтвердить по прочитанному тексту |",
        "| Ограничения | Есть | Есть |"
    ].join("\n");
    const claims = Array.from({length:9}, (_,i) => ({supported:true,
        sourceUrl:urls[i%2]}));
    assert.equal(checkPairedClaimCoverage({task,plan,taskRunResult:run,answer,claims}).valid,true);
    const categorical=answer.replace("Не удалось подтвердить по прочитанному тексту", "Патентных условий нет");
    assert.equal(checkPairedClaimCoverage({task,plan,taskRunResult:run,answer:categorical,claims}).valid,false);
});
