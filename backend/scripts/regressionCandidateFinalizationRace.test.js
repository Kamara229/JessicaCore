import test from "node:test";
import assert from "node:assert/strict";
import {updateCandidateMemory}
  from "../core/learning/candidateMemory/candidateStorage.js";
import {transitionCandidateFinalStatus}
  from "../core/learning/candidateMemory/candidateFinalStatus.js";
import {candidateRaceClient,candidateRaceIdentity}
  from "./fixtures/candidateRaceClient.js";

const {id,revision}=candidateRaceIdentity;
const updatedEvidence={occurrences:2,confidence:0.9,
  examples:[{traceId:"original"},{traceId:"verified-second"}]};

function update(db){
  return updateCandidateMemory({
    id,candidate:updatedEvidence,skillId:"race-skill",
    metadata:{source:"verified",consolidatedFrom:["other"]},
    expectedUpdatedAt:revision,client:db.client
  });
}
function promote(db){
  return transitionCandidateFinalStatus({
    client:db.client,id,status:"PROMOTED",
    metadata:{proposalId:"race-proposal",skillId:"race-skill",
      resolvedAt:"2026-10-27T00:00:00.000Z"}
  });
}

test("finalization commits first: waiting evidence update is fenced out",async()=>{
  const db=candidateRaceClient();
  const pause=db.holdNextUpdate("evidence");
  const inFlight=update(db);
  await pause.entered;
  const final=await promote(db);
  assert.equal(final.success,true);
  assert.equal(final.updated,true);
  pause.release();
  await assert.rejects(inFlight,/changed or finalized/);
  assert.equal(db.writes,1);
  assert.equal(db.row.status,"PROMOTED");
  assert.deepEqual(db.row.candidate_json.examples,
    [{traceId:"original"}]);
  assert.equal(db.row.metadata.proposalId,"race-proposal");
});

test("evidence wins first: stale finalizer fails, retry preserves evidence",async()=>{
  const db=candidateRaceClient();
  const pause=db.holdNextUpdate("finalize");
  const inFlight=promote(db);
  await pause.entered;
  const saved=await update(db);
  assert.equal(saved.status,"ACTIVE");
  assert.equal(saved.occurrences,2);
  pause.release();
  const raced=await inFlight;
  assert.equal(raced.success,false);
  assert.equal(raced.conflict,true);
  assert.equal(db.row.status,"ACTIVE");
  assert.equal(db.writes,1);
  const recovered=await promote(db);
  assert.equal(recovered.success,true);
  assert.equal(recovered.updated,true);
  assert.equal(db.row.status,"PROMOTED");
  assert.equal(db.row.occurrences,2);
  assert.deepEqual(db.row.metadata.consolidatedFrom,["other"]);
  assert.equal(db.row.metadata.proposalId,"race-proposal");
  assert.equal(db.writes,2);
  const again=await promote(db);
  assert.equal(again.existing,true);
  assert.equal(again.updated,false);
  assert.equal(db.writes,2);
});

test("race never enables evidence updates after a final decision",async()=>{
  const db=candidateRaceClient();
  await promote(db);
  for(let attempt=0;attempt<2;attempt++)
    await assert.rejects(update(db),/changed or finalized/);
  assert.equal(db.writes,1);
  assert.equal(db.row.status,"PROMOTED");
});
