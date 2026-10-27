import assert from "node:assert/strict";
import {spawn} from "node:child_process";
import {assertIsolatedCi,sql} from "./fixtures/candidateRacePgClient.mjs";

assertIsolatedCi();
const candidate="aa4f2c21-c621-45db-b71e-33f9a5e8bd10";
const seed=await sql(`
BEGIN;
INSERT INTO public.learning_candidates(id,status) VALUES('${candidate}','ACTIVE');
INSERT INTO public.learning_proposals(id,status,action,analysis,processing_at)
SELECT 'ci-failed-outage-proposal-'||suffix,'PROCESSING','NEW_SKILL',
 jsonb_build_object('candidateMemory',jsonb_build_object('id','${candidate}')),now()
FROM (VALUES ('a'),('b')) AS t(suffix);
INSERT INTO public.jessica_experience_skills(id)
VALUES('ci-failed-outage-skill-a'),('ci-failed-outage-skill-b');
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES('ci-failed-outage-skill-a',1,
 jsonb_build_object('id','ci-failed-outage-skill-a',
 'version',1,'name','Independent Skill a',
 'metadata',jsonb_build_object('proposalId','ci-failed-outage-proposal-a')));
COMMIT;
`);
assert.equal(seed.code,0,seed.stderr);

function worker(suffix,fault="normal"){
  return new Promise((resolve,reject)=>{
    const child=spawn(process.execPath,[
      "supabase/tests/real_competing_proposal_worker_ci.mjs",suffix,fault
    ],{stdio:["ignore","pipe","pipe"],
      env:{...process.env,JESSICA_COMPETING_ISOLATED_SCENARIO:"readback-outage"}});
    let out="",err="";
    child.stdout.on("data",v=>{out+=v;});
    child.stderr.on("data",v=>{err+=v;});
    child.on("error",reject);
    child.on("close",code=>resolve({code,out,err}));
  });
}
async function snapshot(){
  const r=await sql(`
SELECT json_build_object(
 'candidate_status',(SELECT status FROM public.learning_candidates WHERE id='${candidate}'),
 'candidate_revision',(SELECT revision FROM public.learning_candidates WHERE id='${candidate}'),
 'candidate_owner',(SELECT metadata->>'proposalId' FROM public.learning_candidates WHERE id='${candidate}'),
 'claims',(SELECT count(*) FROM public.jessica_candidate_skill_publications WHERE candidate_id='${candidate}'),
 'claim_owner',(SELECT proposal_id FROM public.jessica_candidate_skill_publications WHERE candidate_id='${candidate}'),
 'versions_a',(SELECT count(*) FROM public.jessica_experience_skill_versions WHERE skill_id='ci-failed-outage-skill-a'),
 'versions_b',(SELECT count(*) FROM public.jessica_experience_skill_versions WHERE skill_id='ci-failed-outage-skill-b'),
 'status_a',(SELECT status FROM public.learning_proposals WHERE id='ci-failed-outage-proposal-a'),
 'status_b',(SELECT status FROM public.learning_proposals WHERE id='ci-failed-outage-proposal-b'),
 'approved_a',(SELECT approved_at FROM public.learning_proposals WHERE id='ci-failed-outage-proposal-a'),
 'approved_b',(SELECT approved_at FROM public.learning_proposals WHERE id='ci-failed-outage-proposal-b')
)::text;
`);
  assert.equal(r.code,0,r.stderr);
  return JSON.parse(r.stdout);
}

// Winner was committed in PostgreSQL before workers started. The production
// Approval facade finishes Candidate PROMOTED and Proposal APPROVED exactly once.
const winner=await worker("a");
assert.equal(winner.code,0,winner.err||winner.out);
assert.match(winner.out,/COMPETING_WINNER=a/);

// Losing FAILED status commits, but both the application write ACK and a
// separate PostgreSQL readback connection become unavailable.
const lost=await worker("b","lost-failed-ack-offline");
assert.equal(lost.code,0,lost.err||lost.out);
assert.match(lost.out,/COMPETING_LOSER_UNCERTAIN=b/);
const durable=await snapshot();
assert.equal(durable.candidate_status,"PROMOTED");
assert.equal(durable.candidate_revision,1);
assert.equal(durable.candidate_owner,"ci-failed-outage-proposal-a");
assert.equal(durable.claims,1);
assert.equal(durable.claim_owner,"ci-failed-outage-proposal-a");
assert.equal(durable.versions_a,1);
assert.equal(durable.versions_b,0);
assert.equal(durable.status_a,"APPROVED");
assert.ok(durable.approved_a);
assert.equal(durable.status_b,"FAILED","FAILED really committed despite lost ACK");
assert.equal(durable.approved_b,null);

// Restart eight independent losing workers while status readback remains down.
// Each must defer (not assume FAILED confirmation and not re-publish a Skill).
const deferred=await Promise.all(Array.from({length:8},()=>
  worker("b","readback-offline")));
for(const result of deferred){
  assert.equal(result.code,0,result.err||result.out);
  assert.match(result.out,/COMPETING_LOSER_UNCERTAIN=b/);
}
assert.deepEqual(await snapshot(),durable,
  "unconfirmed terminal status must never change Candidate, Skill or Proposal");

// Restored readback can finally confirm the exact FAILED terminal row. Further
// worker restarts must be idempotent; no lost status may be overwritten.
const recovered=await worker("b");
assert.equal(recovered.code,0,recovered.err||recovered.out);
assert.match(recovered.out,/COMPETING_LOSER_REPLAY=b/);
const another=await Promise.all([worker("a"),worker("b"),worker("b")]);
for(const [i,result] of another.entries()){
  assert.equal(result.code,0,result.err||result.out);
  assert.match(result.out,i===0?/COMPETING_WINNER=a/:/COMPETING_LOSER_REPLAY=b/);
}
assert.deepEqual(await snapshot(),durable,"restored readback is read-only");
console.log("PASS: FAILED COMMIT + lost ACK + readback outage + 8 restarts + recovery");
