import test from "node:test";
import assert from "node:assert/strict";
import {persistExperienceSkill}
  from "../experience/storage/experiencePersistence/experienceWriter.js";
import {checkLosingCandidatePublication}
  from "../core/learning/approval/approvalPublicationConflict.js";
import {parallelSkillPublicationStore}
  from "./fixtures/parallelSkillPublicationStore.js";

function experience(proposalId,candidateId,skillId){
  return {
    id:skillId,name:"Concurrent atomic Skill publication regression",
    version:1,previousVersion:null,mode:"create",enabled:false,
    workflow:["Verify shared Candidate publication ownership"],
    metadata:{proposalId,candidateMemoryId:candidateId}
  };
}

test("32 simultaneous saves of one Proposal commit one immutable version",async()=>{
  const store=parallelSkillPublicationStore({preflightCount:32});
  const same=experience("same-proposal","same-candidate","same-skill");
  const results=await Promise.all(Array.from({length:32},
    ()=>persistExperienceSkill(same,store)));
  assert.ok(results.every(r=>r.success===true));
  assert.ok(results.every(r=>r.skillId==="same-skill"&&r.version===1));
  assert.equal(store.state.commits,1);
  assert.equal(store.state.receipts.size,1);
  assert.equal(store.state.histories.size,1);
  assert.equal(store.state.claims.size,1);
  assert.equal(store.state.maxInFlight,1);
  assert.equal(store.state.saveAttempts,32);
  assert.equal(results.filter(r=>r.existing===false).length,1);
  const replay=await persistExperienceSkill(same,store);
  assert.equal(replay.existing,true);
  assert.equal(store.state.saveAttempts,32);
});

test("32 racing Proposals linked to one Candidate produce one publication owner",async()=>{
  const store=parallelSkillPublicationStore({preflightCount:32});
  const a=experience("race-proposal-a","race-candidate","race-skill-a");
  const b=experience("race-proposal-b","race-candidate","race-skill-b");
  const entries=[...Array(16).fill(a),...Array(16).fill(b)];
  const results=await Promise.all(entries.map(value=>
    persistExperienceSkill(value,store)));
  const owner=store.state.claims.get("race-candidate");
  assert.ok(owner?.proposalId==="race-proposal-a"||
    owner?.proposalId==="race-proposal-b");
  assert.equal(store.state.commits,1);
  assert.equal(store.state.histories.size,1);
  assert.equal(store.state.claims.size,1);
  assert.equal(store.state.maxInFlight,1);
  assert.ok(store.state.conflicts>=1);
  for(let i=0;i<entries.length;i++){
    const result=results[i];
    if(entries[i].metadata.proposalId===owner.proposalId){
      assert.equal(result.success,true);
    }else{
      assert.equal(result.success,false);
      assert.equal(result.uncertain,true);
    }
  }
  const loser=owner.proposalId==="race-proposal-a"?b:a;
  const proof=await checkLosingCandidatePublication({
    proposal:{id:loser.metadata.proposalId,action:"NEW_SKILL",
      candidateMemory:{id:"race-candidate"}},
    readOwner:async()=>({...owner,evidence:"durable-unique-candidate-claim"})
  });
  assert.equal(proof.confirmed,true);
  assert.equal(proof.winningProposalId,owner.proposalId);
});

test("independent Candidate publications do not share a global lock",async()=>{
  const store=parallelSkillPublicationStore({preflightCount:2});
  const [a,b]=await Promise.all([
    persistExperienceSkill(experience("ind-a","candidate-a","skill-a"),store),
    persistExperienceSkill(experience("ind-b","candidate-b","skill-b"),store)
  ]);
  assert.equal(a.success,true);
  assert.equal(b.success,true);
  assert.equal(store.state.commits,2);
  assert.equal(store.state.claims.size,2);
  assert.equal(store.state.histories.size,2);
  assert.equal(store.state.maxInFlight,2);
});

test("lost ACK amid 24 parallel replays is recovered from one durable receipt",async()=>{
  const store=parallelSkillPublicationStore({
    preflightCount:24,ackLoss:true
  });
  const value=experience("ack-proposal","ack-candidate","ack-skill");
  const results=await Promise.all(Array.from({length:24},
    ()=>persistExperienceSkill(value,store)));
  assert.ok(results.every(r=>r.success===true));
  assert.equal(store.state.lostAcks,1);
  assert.equal(store.state.commits,1);
  assert.equal(store.state.claims.size,1);
  assert.equal(store.state.histories.size,1);
  assert.equal(results.filter(r=>r.commitRecovered===true).length,1);
  assert.equal(store.state.saveAttempts,24);
  const replay=await persistExperienceSkill(value,store);
  assert.equal(replay.success,true);
  assert.equal(replay.existing,true);
  assert.equal(store.state.saveAttempts,24);
});
