import assert from "node:assert/strict";
import {spawn} from "node:child_process";
import {assertIsolatedCi,sql,waitFor} from "./fixtures/candidateRacePgClient.mjs";

assertIsolatedCi();
const candidate="f9c31335-997b-4c2f-82fc-755c30b529f3";
const proposal="ci-candidate-lost-ack-proposal";
const skill="ci-candidate-lost-ack-skill";
const seed=await sql(`
BEGIN;
INSERT INTO public.learning_candidates(id,status) VALUES('${candidate}','ACTIVE');
INSERT INTO public.learning_proposals(id,status,action,analysis,processing_at)
 VALUES('${proposal}','PROCESSING','NEW_SKILL',
 jsonb_build_object('candidateMemory',jsonb_build_object('id','${candidate}')),now());
INSERT INTO public.jessica_experience_skills(id) VALUES('${skill}');
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
 VALUES('${skill}',1,jsonb_build_object(
 'id','${skill}','version',1,'name','Crash-window Skill',
 'metadata',jsonb_build_object('proposalId','${proposal}')));
COMMIT;
`);
assert.equal(seed.code,0,seed.stderr);

function worker(phase){
  return new Promise((resolve,reject)=>{
    const child=spawn(process.execPath,
      ["supabase/tests/real_skill_candidate_gap_worker_ci.mjs",phase],
      {stdio:["ignore","pipe","pipe"]});
    let output="",errors="";
    child.stdout.on("data",x=>{output+=x;});
    child.stderr.on("data",x=>{errors+=x;});
    child.on("error",reject);
    child.on("close",code=>resolve({code,output,errors}));
  });
}

// Both fresh workers must contend on the same real PostgreSQL Candidate lock.
const holder=sql(`
BEGIN;
SELECT id FROM public.learning_candidates WHERE id='${candidate}' FOR UPDATE;
SELECT pg_sleep(8);
COMMIT;
`,"jessica-ci-ack-lock-holder");
const holderReady=await waitFor(`
SELECT CASE WHEN EXISTS(SELECT 1 FROM pg_stat_activity
 WHERE application_name='jessica-ci-ack-lock-holder' AND state='active'
 AND query LIKE '%pg_sleep(8)%') THEN 1 ELSE 0 END;
`,3500);
assert.equal(holderReady,true);
const workers=[worker("ack-race"),worker("ack-race")];
const contention=await waitFor(`
SELECT CASE WHEN count(*)>=2 THEN 1 ELSE 0 END FROM pg_stat_activity
WHERE application_name='jessica-ci-candidate-promote'
AND wait_event_type='Lock';
`,5500);
const [lock,...results]=await Promise.all([holder,...workers]);
assert.equal(contention,true,"both workers must block on PostgreSQL Candidate lock");
assert.equal(lock.code,0,lock.stderr);
for(const r of results)assert.equal(r.code,0,r.errors||r.output);
assert.equal(results.filter(r=>r.output.includes("ACK_RACE_DEFERRED")).length,1,
 "the successful Candidate COMMIT must lose exactly one ACK");
assert.equal(results.filter(r=>r.output.includes("ACK_RACE_RECOVERED")).length,1,
 "the competing worker must finish from durable Candidate ownership");

async function snapshot(){
  const r=await sql(`
SELECT json_build_object(
 'candidate_status',(SELECT status FROM public.learning_candidates WHERE id='${candidate}'),
 'candidate_revision',(SELECT revision FROM public.learning_candidates WHERE id='${candidate}'),
 'candidate_owner',(SELECT metadata->>'proposalId' FROM public.learning_candidates
 WHERE id='${candidate}'),
 'proposal_status',(SELECT status FROM public.learning_proposals WHERE id='${proposal}'),
 'approved_at',(SELECT approved_at FROM public.learning_proposals WHERE id='${proposal}'),
 'versions',(SELECT count(*) FROM public.jessica_experience_skill_versions
 WHERE skill_id='${skill}'),
 'claims',(SELECT count(*) FROM public.jessica_candidate_skill_publications
 WHERE candidate_id='${candidate}' AND proposal_id='${proposal}')
)::text;
`);
  assert.equal(r.code,0,r.stderr);
  return JSON.parse(r.stdout);
}
const state=await snapshot();
assert.equal(state.candidate_status,"PROMOTED");
assert.equal(state.candidate_revision,1);
assert.equal(state.candidate_owner,proposal);
assert.equal(state.proposal_status,"APPROVED");
assert.ok(state.approved_at);
assert.equal(state.versions,1);
assert.equal(state.claims,1);

// Another process must see APPROVED without rewriting its timestamp.
const replay=await worker("ack-replay");
assert.equal(replay.code,0,replay.errors||replay.output);
assert.deepEqual(await snapshot(),state);
console.log("PASS: lost Candidate COMMIT ACK, concurrent recovery, one version");
