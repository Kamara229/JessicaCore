import test from "node:test";
import assert from "node:assert/strict";
import {saveApprovedSkillVersion}
  from "../core/learning/approval/approvalSkillSave.js";
import {checkLosingCandidatePublication}
  from "../core/learning/approval/approvalPublicationConflict.js";
import {readCandidatePublicationOwner}
  from "../core/learning/approval/approvalPublicationClaimReader.js";

const candidateId="00000000-0000-4000-8000-000000000991";
function makeProposal(id,skillId){
  return {id,action:"NEW_SKILL",confidence:0.94,
    candidateMemory:{id:candidateId},
    proposedExperience:{id:skillId,name:"Verified evidence"}};
}
const context=p=>({proposal:p,review:{valid:true},quality:{passed:true},
  autonomy:{action:"AUTO_APPROVE",confidence:0.94}});
const loser=makeProposal("second-proposal","different-skill");

test("competing Proposal is terminal FAILED only after durable winner proof",async()=>{
  let published=0,finalized=0,promoted=0;
  const owner={proposalId:"first-proposal",skillId:"original-skill",
    evidence:"atomic-publication-claim"};
  const deps={
    approve:async()=>{published++;return {
      success:false,uncertain:true,stage:"storage",error:"publication RPC refused"
    };},
    checkPublicationConflict:args=>checkLosingCandidatePublication({
      proposal:args.proposal,readOwner:async id=>{
        assert.equal(id,candidateId);return owner;
      }
    }),
    markFailed:async()=>{finalized++;return {
      status:"FAILED",persistence:{success:true},success:false
    };},
    promote:async()=>{promoted++;throw Error("Loser cannot promote");}
  };
  const result=await saveApprovedSkillVersion(context(loser),deps);
  assert.equal(result.success,false);
  assert.equal(result.deferred,false);
  assert.equal(result.status,"FAILED");
  assert.equal(result.publicationConflict,true);
  assert.equal(result.winningProposalId,"first-proposal");
  assert.equal(result.proposalStateUpdated,true);
  assert.equal(published,1);
  assert.equal(finalized,1);
  assert.equal(promoted,0);
});

test("missing or unavailable durable owner cannot terminate ambiguous Proposal",async()=>{
  for(const readOwner of [
    async()=>null,
    async()=>{throw Error("Supabase read timeout");},
    async()=>({proposalId:loser.id,skillId:"different-skill"})
  ]){
    let finalized=0;
    const result=await saveApprovedSkillVersion(context(loser),{
      approve:async()=>({success:false,uncertain:true,error:"ACK lost"}),
      checkPublicationConflict:args=>checkLosingCandidatePublication({
        proposal:args.proposal,readOwner}),
      markFailed:async()=>{finalized++;throw Error("Unexpected terminal write");}
    });
    assert.equal(result.status,"PROCESSING");
    assert.equal(result.deferred,true);
    assert.equal(result.success,false);
    assert.equal(finalized,0);
  }
});

test("confirmed loser whose FAILED status ACK is absent remains recoverable",async()=>{
  const result=await saveApprovedSkillVersion(context(loser),{
    approve:async()=>({success:false,uncertain:true,error:"RPC conflict"}),
    checkPublicationConflict:async()=>({confirmed:true,
      winningProposalId:"winner",reason:"claimed by winner"}),
    markFailed:async()=>({success:false,status:"FAILED",
      persistence:{success:false,error:"status ACK missing"}})
  });
  assert.equal(result.success,false);
  assert.equal(result.deferred,true);
  assert.equal(result.status,"PROCESSING");
  assert.equal(result.publicationConflict,true);
  assert.equal(result.proposalStateUpdated,false);
});

test("non-NEW_SKILL proposal never claims ownership of Candidate",async()=>{
  let reads=0;
  const result=await checkLosingCandidatePublication({
    proposal:{...loser,action:"SKILL_IMPROVEMENT"},
    readOwner:async()=>{reads++;return {proposalId:"winner"};}
  });
  assert.equal(result.confirmed,false);
  assert.equal(reads,0);
});

function mockClient({claim=null,candidate=null,claimError=null}={}){
  const reads=[];
  const client={from(table){
    const query={select(){return this;},eq(key,value){
      assert.equal(key,table==="learning_candidates"?"id":"candidate_id");
      assert.equal(value,candidateId);return this;
    },async maybeSingle(){
      reads.push(table);
      if(table==="jessica_candidate_skill_publications")
        return {data:claim,error:claimError};
      if(table==="learning_candidates")
        return {data:candidate,error:null};
      throw Error("Unexpected table "+table);
    }};
    return query;
  }};
  return {client,reads};
}

test("publication claim reader uses DB winner without consulting Candidate",async()=>{
  const db=mockClient({claim:{candidate_id:candidateId,
    proposal_id:"winner",skill_id:"verified",skill_version:2}});
  const result=await readCandidatePublicationOwner(candidateId,{client:db.client});
  assert.equal(result.proposalId,"winner");
  assert.deepEqual(db.reads,["jessica_candidate_skill_publications"]);
});

test("legacy PROMOTED owner is read only when new claim does not exist",async()=>{
  const db=mockClient({candidate:{status:"PROMOTED",
    metadata:{proposalId:"legacy-winner",skillId:"legacy-skill"}}});
  const owner=await readCandidatePublicationOwner(candidateId,{client:db.client});
  assert.equal(owner.proposalId,"legacy-winner");
  assert.equal(owner.evidence,"legacy-promoted-owner");
  assert.equal(db.reads.length,2);
});

test("claim read error does not become a fabricated losing decision",async()=>{
  const db=mockClient({claimError:{message:"read only outage"}});
  await assert.rejects(
    readCandidatePublicationOwner(candidateId,{client:db.client}),
    /read only outage/
  );
  assert.deepEqual(db.reads,["jessica_candidate_skill_publications"]);
});
