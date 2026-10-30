import test from "node:test";
import assert from "node:assert/strict";
import { applyStrictNumericOnlyFormat } from "../core/response/strictNumericOnlyResponse.js";

const task = "Вычисли 17 × 19. Ответь только числом, без пояснений.";
const complete = text => ({
    success: true, status: "COMPLETED", validated: true,
    answerSource: "composer", text
});

test("reported ordinary-chat incident: only the verified-looking number is rendered", () => {
    const original = complete("17 × 19 = 323.");
    const result = applyStrictNumericOnlyFormat(task, original);
    assert.equal(result.text, "323");
    assert.equal(result.validated, original.validated);
    assert.equal(result.answerSource, original.answerSource);
    assert.equal(original.text, "17 × 19 = 323.");
});

test("number with punctuation is normalized without changing its value", () => {
    assert.equal(applyStrictNumericOnlyFormat(task, complete("323.")).text, "323");
    assert.equal(applyStrictNumericOnlyFormat(task, complete("12,5")).text, "12,5");
});

test("no rewriting if the user did not require numeric-only output", () => {
    const response = complete("17 × 19 = 323.");
    assert.equal(applyStrictNumericOnlyFormat("Вычисли 17 × 19.", response), response);
});

test("do not hide failures, uncertainty, prose, or multi-line answers", () => {
    for (const response of [
        { ...complete("17 × 19 = 323."), status: "PARTIAL" },
        { ...complete("17 × 19 = 323."), success: false },
        complete("Получилось 323"),
        complete("17 × 19 = 323\nНо проверка не выполнена")
    ]) {
        assert.equal(applyStrictNumericOnlyFormat(task, response), response);
    }
});
