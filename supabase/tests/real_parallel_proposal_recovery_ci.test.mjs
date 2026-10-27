import assert from "node:assert/strict";
import {spawn} from "node:child_process";
import {assertIsolatedCi,sql,waitFor} from "./fixtures/candidateRacePgClient.mjs";

assertIsolatedCi();
const candidate="f4ca0fa8-45d0-4656-9bf6-b096a4858cc4";
const proposal="ci-parallel-recovery-proposal";
const skill="ci-parallel-recovery-skill";

const seed=await sql(`
BEGIN;
INSERT INTO public.learning_candidates(id,status)
 VALUES('${candidate}','ACTIVE');
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

function freshWorker(){
  return new Promise((resolve,reject)=>{
    const child=spawn(process.execPath,
      ["supabase/tests/real_skill_candidate_gap_worker_ci.mjs","race"],
      {stdio:["ignore","pipe","pipe"]});
    let output="",errors="";
    child.stdout.on("data",x=>{output+=x;});
    child.stderr.on("data",x=>{errors+=x;});
    child.on("error",reject);
    child.on("close",code=>resolve({code,output,errors}));
  });
}

// Hold a real row lock so both independently started workers race for it.
const holder=sql(`
BEGIN;
SELECT id FROM public.learning_candidates WHERE id='${candidate}' FOR UPDATE;
SELECT pg_sleep(8);
COMMIT;
`,"jessica-ci-parallel-lock-holder");
const holding=await waitFor(`
SELECT CASE WHEN EXISTS(SELECT 1 FROM pg_stat_activity
 WHERE application_name='jessica-ci-parallel-lock-holder'
 AND state='active' AND query LIKE '%pg_sleep(8)%') THEN 1 ELSE 0 END;
`,3500);
assert.equal(holding,true,"the Candidate lock must precede worker startup");

const workerA=freshWorker();
const workerB=freshWorker();
const bothWaiting=await waitFor(`
SELECT CASE WHEN count(*)>=2 THEN 1 ELSE 0 END
FROM pg_stat_activity
WHERE application_name='jessica-ci-candidate-promote'
 AND wait_event_type='Lock';
`,5500);
const [lock,first,second]=await Promise.all([holder,workerA,workerB]);
assert.equal(bothWaiting,true,"both worker processes must contend on PostgreSQL row lock");
assert.equal(lock.code,0,lock.stderr);
for(const result of [first,second]){
  assert.equal(result.code,0,result.errors||result.output);
  assert.match(result.output,/PASS: fresh worker recovered committed Skill gap race/);
}

async function snapshot(){
  const r=await sql(`
SELECT json_build_object(
 'candidate_status',(SELECT status FROM public.learning_candidates WHERE id='${candidate}'),
 'candidate_revision',(SELECT revision FROM public.learning_candidates WHERE id='${candidate}'),
 'candidate_owner',(SELECT metadata->>'proposalId'
    FROM public.learning_candidates WHERE id='${candidate}'),
 'proposal_status',(SELECT status FROM public.learning_proposals WHERE id='${proposal}'),
 'approved_at',(SELECT approved_at FROM public.learning_proposals WHERE id='${proposal}'),
 'skill_versions',(SELECT count(*) FROM public.jessica_experience_skill_versions
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
assert.equal(state.candidate_revision,1,"exactly one real Candidate CAS update");
assert.equal(state.candidate_owner,proposal);
assert.equal(state.proposal_status,"APPROVED");
assert.ok(state.approved_at);
assert.equal(state.skill_versions,1);
assert.equal(state.claims,1);

const replay=await freshWorker();
assert.equal(replay.code,0,replay.errors);
assert.deepEqual(await snapshot(),state,"replay must not overwrite timestamps or versions");
console.log("PASS: two real workers, one Candidate promotion and Proposal approval");
