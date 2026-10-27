import test from "node:test";
import assert from "node:assert/strict";
import {claimPendingLearningItems} from "../core/learning/learningQueueClaim.js";
import {transitionLearningQueueOwned}
  from "../core/learning/learningQueueOwnedStatus.js";
import {recordProposalSaveFailure}
  from "../core/learning/proposalSaveRetry.js";
import {saveLearningProposalOwned}
  from "../core/learning/proposalStorage/proposalOwnedWriter.js";
import {ownerTokenQueueClient}
  from "./fixtures/ownerTokenQueueClient.js";

const proposal={
  id:"lease-proposal",queueItemId:"lease-item",status:"PENDING_APPROVAL",
  action:"NEW_SKILL",source:"learning_queue",confidence:0.91,
  targetSkill:{id:"new-lease-skill"},
  proposedExperience:{id:"new-lease-skill",name:"Verification",workflow:["check"]},
  analysis:{source:"regression"}
};
async function claim(store){
  const result=await claimPendingLearningItems({client:store.client});
  assert.equal(result.success,true);
  assert.equal(result.items.length,1);
  return result.items[0].claimToken;
}

test("stale Worker cannot alter Proposal, retry count or Queue after recovery",async()=>{
  const store=ownerTokenQueueClient();
  const stale=await claim(store);
  store.recover(); // same transition as stale recovery: token invalidated
  const current=await claim(store);
  assert.notEqual(stale,current);

  const oldStatus=await transitionLearningQueueOwned({
    queueItemId:store.row.id,claimToken:stale,status:"FAILED",
    client:store.client
  });
  const oldRetry=await recordProposalSaveFailure({
    queueItemId:store.row.id,claimToken:stale,client:store.client
  });
  const oldProposal=await saveLearningProposalOwned(proposal,{
    claimToken:stale,client:store.client
  });
  assert.equal(oldStatus.success,false);
  assert.equal(oldStatus.stale,true);
  assert.equal(oldRetry.status,"stale_claim");
  assert.equal(oldRetry.success,false);
  assert.equal(oldProposal.success,false);
  assert.equal(oldProposal.stale,true);
  assert.equal(store.proposals.size,0);
  assert.equal(store.retries,0);
  assert.equal(store.row.status,"PROCESSING");
  assert.equal(store.row.claim_token,current);

  const retry=await recordProposalSaveFailure({
    queueItemId:store.row.id,claimToken:current,client:store.client
  });
  assert.equal(retry.status,"retry");
  assert.equal(retry.attempts,1);
  assert.equal(store.retries,1);
  assert.equal(store.row.status,"PENDING");
  assert.equal(store.row.claim_token,null);

  const third=await claim(store);
  assert.notEqual(current,third);
  const late=await saveLearningProposalOwned(proposal,{
    claimToken:current,client:store.client
  });
  assert.equal(late.success,false);
  const saved=await saveLearningProposalOwned(proposal,{
    claimToken:third,client:store.client
  });
  assert.equal(saved.success,true);
  assert.equal(saved.existing,false);
  assert.equal(saved.proposal.id,proposal.id);
  assert.equal(store.proposals.size,1);
  const final=await transitionLearningQueueOwned({
    queueItemId:store.row.id,claimToken:third,status:"PROPOSED",
    client:store.client
  });
  assert.equal(final.success,true);
  assert.equal(store.row.status,"PROPOSED");
  assert.equal(store.row.claim_token,null);
  assert.equal(store.retries,1);
});

test("missing and mismatched ownership fail closed even for a valid Proposal",async()=>{
  const store=ownerTokenQueueClient();
  const current=await claim(store);
  assert.equal((await transitionLearningQueueOwned({
    queueItemId:store.row.id,status:"PROPOSED",client:store.client
  })).success,false);
  assert.equal((await saveLearningProposalOwned(proposal,{
    claimToken:null,client:store.client
  })).success,false);
  const old="00000000-0000-4000-8000-000000000001";
  assert.equal((await recordProposalSaveFailure({
    queueItemId:store.row.id,claimToken:old,client:store.client
  })).status,"stale_claim");
  assert.equal(store.row.claim_token,current);
  assert.equal(store.row.status,"PROCESSING");
  assert.equal(store.proposals.size,0);
  assert.equal(store.retries,0);
});

test("a failed atomic claim response missing owner token is rejected",async()=>{
  const result=await claimPendingLearningItems({client:{
    async rpc(){return {data:[{
      id:"legacy",status:"PROCESSING",action:"NEW_SKILL"
    }],error:null};}
  }});
  assert.equal(result.success,false);
  assert.deepEqual(result.items,[]);
  assert.match(result.error,/owner token/);
});
