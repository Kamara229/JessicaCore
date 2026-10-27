import assert from "node:assert/strict";
import {spawn} from "node:child_process";
import {assertIsolatedCi,sql} from "./fixtures/candidateRacePgClient.mjs";

assertIsolatedCi();
const candidate="bea2b33c-1a6a-42fd-9693-61cbd2c865b0";
const seed=await sql(`
BEGIN;
INSERT INTO public.learning_candidates(id,status)
VALUES ('${candidate}','ACTIVE');
INSERT INTO public.learning_proposals(id,status,action,analysis,processing_at)
SELECT 'ci-owner-outage-proposal-'||suffix,'PROCESSING','NEW_SKILL',
 jsonb_build_object('candidateMemory',jsonb_build_object('id','${candidate}')),now()
FROM (VALUES ('a'),('b')) AS t(suffix);
INSERT INTO public.jessica_experience_skills(id)
VALUES ('ci-owner-outage-skill-a'),('ci-owner-outage-skill-b');
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES ('ci-owner-outage-skill-a',1,jsonb_build_object(
 'id','ci-owner-outage-skill-a','name','Independent Skill a','version',1,
 'metadata',jsonb_build_object('proposalId','ci-owner-outage-proposal-a')));
COMMIT;
`);
assert.equal(seed.code,0,seed.stderr);

function runWorker(suffix,fault="normal"){
  return new Promise((resolve,reject)=>{
    const child=spawn(process.execPath,[
      "supabase/tests/real_competing_proposal_worker_ci.mjs",suffix,fault
    ],{stdio:["ignore","pipe","pipe"],
      env:{...process.env,JESSICA_COMPETING_ISOLATED_SCENARIO:"owner-readback-outage"}});
    let out="",err="";
    child.stdout.on("data",v=>{out+=v;});
    child.stderr.on("data",v=>{err+=v;});
    child.on("error",reject);
    child.on("close",code=>resolve({code,out,err}));
  });
}

async function state(){
  const r=await sql(`
SELECT json_build_object(
 'candidate_status',(SELECT status FROM public.learning_candidates WHERE id='${candidate}'),
 'candidate_revision',(SELECT revision FROM public.learning_candidates WHERE id='${candidate}'),
 'candidate_owner',(SELECT metadata->>'proposalId' FROM public.learning_candidates WHERE id='${candidate}'),
 'claim_count',(SELECT count(*) FROM public.jessica_candidate_skill_publications WHERE candidate_id='${candidate}'),
 'claim_owner',(SELECT proposal_id FROM public.jessica_candidate_skill_publications WHERE candidate_id='${candidate}'),
 'version_a',(SELECT count(*) FROM public.jessica_experience_skill_versions WHERE skill_id='ci-owner-outage-skill-a'),
 'version_b',(SELECT count(*) FROM public.jessica_experience_skill_versions WHERE skill_id='ci-owner-outage-skill-b'),
 'status_a',(SELECT status FROM public.learning_proposals WHERE id='ci-owner-outage-proposal-a'),
 'status_b',(SELECT status FROM public.learning_proposals WHERE id='ci-owner-outage-proposal-b'),
 'approved_a',(SELECT approved_at FROM public.learning_proposals WHERE id='ci-owner-outage-proposal-a'),
 'approved_b',(SELECT approved_at FROM public.learning_proposals WHERE id='ci-owner-outage-proposal-b'),
 'processing_b',(SELECT processing_at FROM public.learning_proposals WHERE id='ci-owner-outage-proposal-b')
)::text;
`);
  assert.equal(r.code,0,r.stderr);
  return JSON.parse(r.stdout);
}

const winner=await runWorker("a");
assert.equal(winner.code,0,winner.err||winner.out);
assert.match(winner.out,/COMPETING_WINNER=a/);

const unknown=await runWorker("b","owner-offline");
assert.equal(unknown.code,0,unknown.err||unknown.out);
assert.match(unknown.out,/COMPETING_OWNER_UNKNOWN=b/);
const pending=await state();
assert.equal(pending.candidate_status,"PROMOTED");
assert.equal(pending.candidate_revision,1);
assert.equal(pending.candidate_owner,"ci-owner-outage-proposal-a");
assert.equal(pending.claim_count,1);
assert.equal(pending.claim_owner,pending.candidate_owner);
assert.equal(pending.version_a,1);
assert.equal(pending.version_b,0);
assert.equal(pending.status_a,"APPROVED");
assert.ok(pending.approved_a);
assert.equal(pending.status_b,"PROCESSING","unverified loser must stay PROCESSING");
assert.equal(pending.approved_b,null);
assert.ok(pending.processing_b);

// Each restarted worker still sees a denied Skill insert, but the owner read
// fails independently; none may guess the winner or set FAILED.
const repeated=await Promise.all(Array.from({length:8},()=>
  runWorker("b","owner-offline")));
for(const response of repeated){
  assert.equal(response.code,0,response.err||response.out);
  assert.match(response.out,/COMPETING_OWNER_UNKNOWN=b/);
}
assert.deepEqual(await state(),pending,
  "owner-readback outage must not mutate immutable history or Proposal state");

// After the independent owner connection is restored, the durable claim is
// sufficient to terminalize exactly the losing Proposal through real CAS.
const recovered=await runWorker("b");
assert.equal(recovered.code,0,recovered.err||recovered.out);
assert.match(recovered.out,/COMPETING_LOSER=b/);
const final=await state();
assert.deepEqual({...final,status_b:"PROCESSING",processing_b:pending.processing_b},pending);
assert.equal(final.status_b,"FAILED");
assert.equal(final.processing_b,null);

const replays=await Promise.all(Array.from({length:6},()=>
  runWorker("b")));
for(const response of replays){
  assert.equal(response.code,0,response.err||response.out);
  assert.match(response.out,/COMPETING_LOSER_REPLAY=b/);
}
assert.deepEqual(await state(),final,
  "terminal FAILED and ownership must remain immutable on retry");
console.log("PASS: 9 owner-readback failures -> no false FAILED, then CAS finalizes once");
