import test from "node:test";
import assert from "node:assert/strict";
import {randomUUID} from "node:crypto";
import {claimPendingLearningItems} from "../core/learning/learningQueueClaim.js";
import {applyCandidateObservation}
  from "../core/learning/candidateMemory/candidateObservationJournal.js";
import {observeLearningCandidate}
  from "../core/learning/learningCandidateMemory.js";
import {ownedCandidateJournalClient}
  from "./fixtures/ownedCandidateJournalClient.js";

async function claim(store){
  const result=await claimPendingLearningItems({client:store.client});
  assert.equal(result.success,true);
  assert.equal(result.items.length,1);
  return result.items[0].claimToken;
}
const candidate={
  skillId:"skill-candidate-fence",name:"Check a source",
  occurrences:1,confidence:0.9,examples:[]
};
const observed=(token,id)=>{
  return {queueItemId:"candidate-fence-queue",claimToken:token,id,
    candidateKey:"skill:skill-candidate-fence",action:"NEW",
    candidate:{...candidate},metadata:{source:"test"}};
};

test("stale Worker after recovery cannot create Candidate or journal",async()=>{
  const store=ownedCandidateJournalClient();
  const first=await claim(store);
  store.recover();
  const current=await claim(store);
  assert.notEqual(current,first);
  const firstAttempt=applyCandidateObservation({
    ...observed(first,randomUUID()),client:store.client
  });
  const secondAttempt=applyCandidateObservation({
    ...observed(current,randomUUID()),client:store.client
  });
  const outcomes=await Promise.allSettled([firstAttempt,secondAttempt]);
  assert.equal(outcomes[0].status,"rejected");
  assert.match(outcomes[0].reason.message,/Stale Candidate Memory/);
  assert.equal(outcomes[1].status,"fulfilled");
  assert.equal(outcomes[1].value.applied,true);
  assert.equal(store.candidates.size,1);
  assert.equal(store.journal.size,1);
  assert.equal(store.writes,1);
  assert.equal([...store.candidates.values()][0].occurrences,1);
});

test("stale Worker cannot increase occurrences after current Worker committed",async()=>{
  const store=ownedCandidateJournalClient();
  const old=await claim(store);
  const id=randomUUID();
  await applyCandidateObservation({...observed(old,id),client:store.client});
  store.recover();
  const current=await claim(store);
  const before=store.candidates.get(id).occurrences;
  const stale=applyCandidateObservation({
    ...observed(old,id),action:"EXACT",expectedOccurrences:before,
    candidate:{...candidate,occurrences:99},client:store.client
  });
  const valid=applyCandidateObservation({
    ...observed(current,id),action:"EXACT",expectedOccurrences:before,
    candidate:{...candidate,occurrences:2},client:store.client
  });
  const results=await Promise.allSettled([stale,valid]);
  assert.equal(results[0].status,"rejected");
  assert.equal(results[1].status,"fulfilled");
  assert.equal(results[1].value.existing,true,
    "same Queue event has already been recorded; no second merge");
  assert.equal(store.candidates.get(id).occurrences,1);
  assert.equal(store.journal.size,1);
  assert.equal(store.writes,1);
});

test("Candidate observation facade requires a current claim token",async()=>{
  const store=ownedCandidateJournalClient();
  await assert.rejects(applyCandidateObservation({
    ...observed("",randomUUID()),client:store.client
  }),/current Queue claim token/);
  assert.equal(store.writes,0);
});

test("all real Candidate observation paths pass the claim token",async()=>{
  const store=ownedCandidateJournalClient(),token=await claim(store);
  const task={
    id:"candidate-fence-queue",claimToken:token,action:"NEW_SKILL",
    event:{traceId:"trace-owner",payload:{skillCandidate:candidate}}
  };
  const observe=()=>observeLearningCandidate(task,{
    findCandidate:async()=>null,
    listCandidates:async()=>[],
    applyObservation:p=>applyCandidateObservation({...p,client:store.client})
  });
  const initial=await observe();
  const again=await observe();
  assert.equal(initial.observed,true);
  assert.equal(again.observed,false);
  assert.equal(initial.memory.id,again.memory.id);
  assert.equal(store.writes,1);
  store.recover();
  await claim(store);
  await assert.rejects(observe(),/Stale Candidate Memory/);
  assert.equal(store.writes,1);
});
