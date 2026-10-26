import test from "node:test";
import assert from "node:assert/strict";
import {requireExperienceHistoryRows}
 from "../experience/storage/experiencePersistence/experienceHistoryRows.js";
import {resolveExperienceVersionState}
 from "../experience/storage/experiencePersistence/experienceVersioning.js";

test("empty but valid history permits first Skill version",()=>{
    assert.deepEqual(requireExperienceHistoryRows([]),[]);
    const state=resolveExperienceVersionState(requireExperienceHistoryRows([]));
    assert.equal(state.nextVersion,1);
});

test("invalid, partial or silently dropped history is rejected",()=>{
    for(const history of [null,undefined,{},false,"",[{version:1},null],
        [{version:1},undefined],[{version:1},[]]]){
        assert.throws(()=>requireExperienceHistoryRows(history),
            /Experience history result/);
    }
});

test("valid complete history remains eligible for next Skill version",()=>{
    const history=[{version:3},{version:2},{version:1}];
    assert.equal(resolveExperienceVersionState(
        requireExperienceHistoryRows(history)).nextVersion,4);
});
