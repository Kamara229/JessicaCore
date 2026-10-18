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
