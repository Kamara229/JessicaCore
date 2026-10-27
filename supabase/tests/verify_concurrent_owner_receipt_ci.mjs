import assert from "node:assert/strict";
import {readCandidatePublicationOwner}
  from "../../backend/core/learning/approval/approvalPublicationClaimReader.js";
import {checkLosingCandidatePublication}
  from "../../backend/core/learning/approval/approvalPublicationConflict.js";
import {saveApprovedSkillVersion}
  from "../../backend/core/learning/approval/approvalSkillSave.js";
import {candidateOwnerRealPgClient} from "./fixtures/candidateOwnerRealPgClient.mjs";

const client=(id,tamper=null)=>
  candidateOwnerRealPgClient(id,{tamper});
const proposal=(id,candidateId)=>({
  id,action:"NEW_SKILL",confidence:0.95,
  candidateMemory:{id:candidateId},
  proposedExperience:{id:"unpublished",name:"Unpublished contender"}
});
const readOwner=tamper=>id=>
  readCandidatePublicationOwner(id,{client:client(id,tamper)});

export async function assertOwnerNotYetCommitted(candidateId){
  const owner=await readCandidatePublicationOwner(candidateId,{
    client:client(candidateId)
  });
  assert.equal(owner,null,"ACTIVE Candidate without a claim is not evidence");
}

export async function verifyOwnerReceiptAfterRealRace({candidateId,winner,loser}){
  const winningProposal="ci-competing-proposal-"+winner;
  const losing=proposal("ci-competing-proposal-"+loser,candidateId);

  const durable=await readCandidatePublicationOwner(candidateId,{
    client:client(candidateId)
  });
  assert.deepEqual(durable,{
    proposalId:winningProposal,
    skillId:"ci-competing-skill-"+winner,
    evidence:"atomic-publication-claim"
  });

  const verified=await checkLosingCandidatePublication({
    proposal:losing,readOwner:readOwner(null)
  });
  assert.equal(verified.confirmed,true);
  assert.equal(verified.winningProposalId,winningProposal);
  assert.equal(verified.evidence,"atomic-publication-claim");

  // Even if concurrent readback delivered a response for another Candidate
  // or omitted a mandatory receipt field, the loser must stay recoverable.
  for(const tamper of ["wrong-candidate","missing-skill","invalid-version"]){
    const uncertain=await checkLosingCandidatePublication({
      proposal:losing,readOwner:readOwner(tamper)
    });
    assert.equal(uncertain.confirmed,false,tamper);
    assert.equal(uncertain.uncertain,true,tamper);
    let terminalWrites=0;
    const result=await saveApprovedSkillVersion({
      proposal:losing,review:{valid:true},quality:{passed:true},
      autonomy:{action:"AUTO_APPROVE",confidence:0.95}
    },{
      approve:async()=>({success:false,uncertain:true,error:"duplicate claim"}),
      checkPublicationConflict:({proposal:p})=>
        checkLosingCandidatePublication({proposal:p,readOwner:readOwner(tamper)}),
      markFailed:async()=>{
        terminalWrites++;
        throw Error("corrupt receipt cannot finalize a Proposal");
      }
    });
    assert.equal(result.success,false);
    assert.equal(result.status,"PROCESSING");
    assert.equal(result.deferred,true);
    assert.equal(result.proposalStateUpdated,false);
    assert.equal(terminalWrites,0);
  }
  console.log("PASS: 2-worker Candidate winner receipt verified; 3 corrupt readbacks fail closed");
}
