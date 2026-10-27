import test from "node:test";
import assert from "node:assert/strict";
import {validatedCandidateOwnerReceipt}
  from "../core/learning/approval/approvalOwnerReceipt.js";
import {readCandidatePublicationOwner}
  from "../core/learning/approval/approvalPublicationClaimReader.js";
import {checkLosingCandidatePublication}
  from "../core/learning/approval/approvalPublicationConflict.js";

const candidateId="00000000-0000-4000-8000-000000000991";
const valid={candidate_id:candidateId,proposal_id:"other",
  skill_id:"skill-1",skill_version:1};

function claimClient(claim){
  return {from(table){
    assert.equal(table,"jessica_candidate_skill_publications");
    return {select(){return this;},eq(k,v){
      assert.equal(k,"candidate_id");assert.equal(v,candidateId);return this;
    },async maybeSingle(){return {data:claim,error:null};}};
  }};
}

test("exact durable Candidate, Proposal, Skill and version form an owner receipt",()=>{
  assert.deepEqual(validatedCandidateOwnerReceipt(candidateId,valid),{
    proposalId:"other",skillId:"skill-1",evidence:"atomic-publication-claim"
  });
});

test("wrong Candidate or incomplete receipt is never ownership proof",async()=>{
  for(const bad of [
    {...valid,candidate_id:"00000000-0000-4000-8000-000000000992"},
    {...valid,proposal_id:""},
    {...valid,skill_id:null},
    {...valid,skill_version:0},
    {...valid,skill_version:1.5},
    {...valid,skill_version:null}
  ]){
    const readOwner=id=>readCandidatePublicationOwner(id,{
      client:claimClient(bad)});
    await assert.rejects(readOwner(candidateId),/receipt is inconsistent/);
    const result=await checkLosingCandidatePublication({
      proposal:{id:"current",action:"NEW_SKILL",
        candidateMemory:{id:candidateId}},readOwner
    });
    assert.equal(result.confirmed,false);
    assert.equal(result.uncertain,true);
    assert.equal(result.winningProposalId,undefined);
  }
});

test("a matching receipt can confirm a real losing Proposal",async()=>{
  const result=await checkLosingCandidatePublication({
    proposal:{id:"current",action:"NEW_SKILL",
      candidateMemory:{id:candidateId}},
    readOwner:id=>readCandidatePublicationOwner(id,{
      client:claimClient(valid)})
  });
  assert.equal(result.confirmed,true);
  assert.equal(result.winningProposalId,"other");
  assert.equal(result.evidence,"atomic-publication-claim");
});
