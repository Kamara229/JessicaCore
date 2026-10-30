import test from "node:test";
import assert from "node:assert/strict";
import { groundedApachePatentSection } from "../core/composer/groundedApachePatentSection.js";

const url = "https://www.apache.org/licenses/LICENSE-2.0.txt";
const task = "Найди официальный текст Apache License 2.0. В каком разделе указано прекращение патентной лицензии при подаче патентного иска? Укажи номер раздела и ссылку";
const plan = {
    intent: "native_single_license_primary_text",
    metadata: { licenseId: "Apache-2.0" },
    steps: [{ tool: "web_fetch", arguments: { url } }]
};
const source = [
    "Apache License Version 2.0",
    "3. Grant of Patent License. Each Contributor hereby grants to You a perpetual,",
    "worldwide, non-exclusive, royalty-free patent license to make, have made, use.",
    "If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit)",
    "alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement,",
    "then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.",
    "4. Redistribution."
].join(" ");

function run(content = source, truncated = false) {
    return { results: [{ tool: "web_fetch", success: true, data: { url, content, truncated } }] };
}

test("exact owner report must name the checked Apache section, conditions and official link", () => {
    const result = groundedApachePatentSection(task, plan, run());
    assert.equal(result?.success, true);
    assert.match(result.text, /Раздел 3 — «Grant of Patent License»/u);
    assert.match(result.text, /иска/u);
    assert.match(result.text, /с даты подачи иска/u);
    assert.ok(result.text.includes(url));
});

test("no section claim without complete verified source or from truncated fetch", () => {
    assert.equal(groundedApachePatentSection(task, plan, run(source.replace("3. Grant", "5. Grant"))), null);
    assert.equal(groundedApachePatentSection(task, plan, run(source, true)), null);
});
