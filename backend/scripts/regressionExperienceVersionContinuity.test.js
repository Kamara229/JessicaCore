import test from "node:test";
import assert from "node:assert/strict";
import {resolveExperienceVersionState} from "../experience/storage/experiencePersistence/experienceVersioning.js";
import {resolveLearningSaveMode} from "../experience/learning/learningSaveMode.js";

test("NEW_SKILL starts v1 while SKILL_IMPROVEMENT advances without overwriting history", () => {
    const first = resolveExperienceVersionState([]);
    assert.deepEqual(first, {exists:false,latestVersion:null,nextVersion:1,previousVersion:null});
    assert.equal(resolveLearningSaveMode(first),"create");
    const after = resolveExperienceVersionState([{version:1}]);
    assert.deepEqual(after, {exists:true,latestVersion:1,nextVersion:2,previousVersion:1});
    assert.equal(resolveLearningSaveMode(after),"update");
    const many = resolveExperienceVersionState(Array.from({length:17},(_,i)=>({version:17-i})));
    assert.equal(many.nextVersion,18);
    assert.equal(resolveLearningSaveMode(many),"update");
});

test("missing, duplicate, invalid snapshots fail closed instead of silently skipping versions", () => {
    for (const history of [
        null, undefined, {}, "invalid-history",
        [{version:1},{version:3}],
        [{version:2}],
        [{version:1},{version:1}],
        [{version:1},{}],
        [{version:1},{version:0}],
        [{version:1},{version:2.5}]
    ]) {
        assert.throws(() => resolveExperienceVersionState(history),/Experience history/);
    }
});
