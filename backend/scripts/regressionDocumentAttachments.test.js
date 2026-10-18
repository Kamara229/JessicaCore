import test from "node:test";
import assert from "node:assert/strict";
import { normalizeAttachmentContexts, attachFilesToDecomposition } from "../core/files/attachmentContext.js";

test("file extract reaches execution even after task decomposition", () => {
    const source = {subtasks: [{id: 1, text: "Сравни документы", dependsOn: []}]};
    const result = attachFilesToDecomposition(source, [{name:"test.txt",text:"В документе указано СИРИУС-47"}]);
    assert.equal(result.subtasks[0].id, 1);
    assert.deepEqual(result.subtasks[0].dependsOn, []);
    assert.match(result.subtasks[0].text, /СИРИУС-47/);
    assert.doesNotMatch(source.subtasks[0].text, /СИРИУС/);
});
test("limits large files, number of files, and unsafe object fields", () => {
    const many = Array.from({length:6}, (_,i) => ({name:"f"+i+".txt", text:"a".repeat(11000)}));
    const result = normalizeAttachmentContexts(many);
    assert.ok(result.length <= 3);
    assert.ok(result.reduce((sum, x) => sum + x.text.length, 0) <= 16000);
    assert.equal(normalizeAttachmentContexts([null, {}, {name:"x",text:""}]).length,0);
});
test("absent attachments leave decomposition untouched", () => {
    const decomposition = {subtasks:[{text:"Hello"}]};
    assert.equal(attachFilesToDecomposition(decomposition, []), decomposition);
});
