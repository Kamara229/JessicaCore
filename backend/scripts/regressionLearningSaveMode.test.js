import test from "node:test";
import assert from "node:assert/strict";
import { resolveLearningSaveMode } from "../experience/learning/learningSaveMode.js";

test("first Experience version must be create", () => {
    assert.equal(resolveLearningSaveMode({exists:false,nextVersion:1,previousVersion:null}),"create");
});

test("existing Experience versions must be update even for duplicate NEW_SKILL", () => {
    assert.equal(resolveLearningSaveMode({exists:true,nextVersion:2,previousVersion:1}),"update");
    assert.equal(resolveLearningSaveMode({exists:true,nextVersion:18,previousVersion:17}),"update");
});

test("broken or ambiguous version state cannot be saved", () => {
    for (const state of [null,{}, {exists:true,nextVersion:3,previousVersion:1},
        {exists:false,nextVersion:2,previousVersion:1},
        {exists:true,nextVersion:1,previousVersion:null}]) {
        assert.throws(()=>resolveLearningSaveMode(state),/Invalid Experience version state/);
    }
});
