import test from "node:test";
import assert from "node:assert/strict";
import {randomUUID} from "node:crypto";
import {recoverStaleLearningQueue}
  from "../core/learning/learningQueueRecovery.js";
import {observeLearningCandidate}
  from "../core/learning/learningCandidateMemory.js";
import {applyCandidateObservation}
  from "../core/learning/candidateMemory/candidateObservationJournal.js";
import {createLearningProposalFromQueue}
  from "../experience/learning/learningProposal.js";
import {checkLearningWorkerReplay}
  from "../core/learning/workerReplayGuard.js";

// Mock only persistence: all Candidate, Proposal and Replay logic is real.
function fakePersistence(){
  const journal=new Map(),candidates=new Map(),proposals=new Map();
  let candidateWrites=0,proposalInserts=0;
  const client={async rpc(name,p){
    assert.equal(name,"apply_candidate_memory_observation_owned");
    assert.equal(p.p_claim_token,"00000000-0000-4000-8000-000000000002");
    const old=journal.get(p.p_queue_item_id);
    if(old){
      return {data:{
        applied:false,existing:true,candidate:structuredClone(candidates.get(old))
      },error:null};
    }
    const id=p.p_candidate_id, candidate=structuredClone(p.p_candidate);
    const row={
      id,candidate_key:p.p_candidate_key,action:"NEW_SKILL",
      skill_id:p.p_skill_id,category:p.p_category,status:"ACTIVE",
      occurrences:candidate.occurrences,confidence:candidate.confidence,
      candidate_json:candidate,metadata:p.p_metadata
    };
    candidates.set(id,row);
    journal.set(p.p_queue_item_id,id);
    candidateWrites++;
    return {data:{applied:true,existing:false,candidate:structuredClone(row)},error:null};
  }};
  return {
    client,journal,candidates,proposals,
    get candidateWrites(){return candidateWrites;},
    get proposalInserts(){return proposalInserts;},
    saveProposal(proposal){
      const existing=proposals.get(proposal.queueItemId);
      if(existing)return {existing:true,proposal:existing};
      proposals.set(proposal.queueItemId,proposal);
      proposalInserts++;
      return {existing:false,proposal};
    }
  };
}

const skill={
  id:"disposable-gap-skill",
  name:"Verify source integrity",
  category:"research",occurrences:1,confidence:0.84,
  workflow:["read source","verify result"],
  examples:[{task:"verify a document",result:"ok",traceId:"gap-test"}]
};
const item={
  id:"queue-post-candidate-crash",claimToken:"00000000-0000-4000-8000-000000000002",action:"NEW_SKILL",
  event:{traceId:"trace-post-candidate-crash",action:"NEW_SKILL",
    confidence:0.84,payload:{skillCandidate:skill}}
};

test("crash after Candidate commit resumes without duplicate observation or Proposal",async()=>{
  const db=fakePersistence();
  const observe=()=>observeLearningCandidate(item,{
    findCandidate:async()=>null,
    listCandidates:async()=>[],
    applyObservation:params=>applyCandidateObservation({...params,client:db.client})
  });

  // First Worker wrote Candidate and died before it could create a Proposal.
  const beforeCrash=await observe();
  assert.equal(beforeCrash.created,true);
  assert.equal(db.candidateWrites,1);
  assert.equal(db.proposals.size,0);

  // Supabase stale-recovery RPC marks only a journal-backed row safe to retry.
  const recovery=await recoverStaleLearningQueue({client:{
    async rpc(name){
      assert.equal(name,"recover_stale_learning_queue");
      assert.equal(db.journal.has(item.id),true);
      return {data:[{id:item.id,action:item.action,status:"PENDING"}],error:null};
    }
  }});
  assert.equal(recovery.success,true);
  assert.equal(recovery.retried,1);
  assert.equal(recovery.quarantined,0);

  // A retry may suggest a different Candidate ID; the journal is authoritative.
  const resumed=await observe();
  assert.equal(resumed.success,true);
  assert.equal(resumed.created,false);
  assert.equal(resumed.observed,false);
  assert.equal(resumed.memory.id,beforeCrash.memory.id);
  assert.equal(resumed.candidate.occurrences,1);
  assert.equal(db.candidateWrites,1);

  const firstProposal=createLearningProposalFromQueue(resumed.queueItem);
  const saved=db.saveProposal(firstProposal);
  assert.equal(saved.existing,false);
  assert.equal(saved.proposal.targetSkill.id,skill.id);
  assert.equal(db.proposalInserts,1);

  // Simulate a lost ACK after Proposal commit: the same queue ID is idempotent.
  const secondProposal=createLearningProposalFromQueue(resumed.queueItem);
  assert.notEqual(secondProposal.id,firstProposal.id);
  const stored=db.saveProposal(secondProposal);
  assert.equal(stored.existing,true);
  assert.equal(stored.proposal.id,firstProposal.id);
  assert.equal(db.proposalInserts,1);

  const queueStatus=[];
  const replay=await checkLearningWorkerReplay(item,{
    findProposal:async id=>({success:true,found:true,proposal:db.proposals.get(id)}),
    updateQueue:async(id,status)=>{queueStatus.push([id,status]);return {success:true};}
  });
  assert.equal(replay.handled,true);
  assert.equal(replay.result.existing,true);
  assert.equal(replay.result.proposalId,firstProposal.id);
  assert.deepEqual(queueStatus,[[item.id,"PROPOSED"]]);
  assert.equal(db.candidates.size,1);
  assert.equal(db.journal.size,1);
  assert.equal(db.proposals.size,1);
});
