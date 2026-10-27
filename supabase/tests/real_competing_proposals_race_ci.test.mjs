import assert from "node:assert/strict";
import {spawn} from "node:child_process";
import {assertIsolatedCi,sql,waitFor} from "./fixtures/candidateRacePgClient.mjs";

assertIsolatedCi();
const candidate="d7d26169-ef7b-4d89-ad79-61b7fb22c007";
const seed=await sql(`
BEGIN;
INSERT INTO public.learning_candidates(id,status) VALUES('${candidate}','ACTIVE');
INSERT INTO public.learning_proposals(id,status,action,analysis,processing_at)
SELECT 'ci-competing-proposal-'||suffix,'PROCESSING','NEW_SKILL',
 jsonb_build_object('candidateMemory',jsonb_build_object('id','${candidate}')),now()
FROM (VALUES ('a'),('b')) AS t(suffix);
INSERT INTO public.jessica_experience_skills(id)
VALUES('ci-competing-skill-a'),('ci-competing-skill-b');
COMMIT;
`);
assert.equal(seed.code,0,seed.stderr);

function worker(suffix){
  return new Promise((resolve,reject)=>{
    const p=spawn(process.execPath,
      ["supabase/tests/real_competing_proposal_worker_ci.mjs",suffix],
      {stdio:["ignore","pipe","pipe"]});
    let out="",err="";
    p.stdout.on("data",x=>{out+=x;});
    p.stderr.on("data",x=>{err+=x;});
    p.on("error",reject);
    p.on("close",code=>resolve({code,out,err}));
  });
}
const blocker=sql(`
BEGIN;
SELECT id FROM public.learning_candidates WHERE id='${candidate}' FOR UPDATE;
SELECT pg_sleep(8);
COMMIT;
`,"jessica-ci-competing-lock-holder");
const held=await waitFor(`
SELECT CASE WHEN EXISTS(SELECT 1 FROM pg_stat_activity
 WHERE application_name='jessica-ci-competing-lock-holder'
 AND state='active' AND query LIKE '%pg_sleep(8)%') THEN 1 ELSE 0 END;
`,3500);
assert.equal(held,true,"Candidate row must be locked before two workers start");

const [a,b]=[worker("a"),worker("b")];
const blocked=await waitFor(`
SELECT CASE WHEN count(*)=2 THEN 1 ELSE 0 END FROM pg_stat_activity
WHERE application_name IN ('jessica-ci-competing-insert-a','jessica-ci-competing-insert-b')
 AND wait_event_type='Lock';
`,5500);
const [hold,resultA,resultB]=await Promise.all([blocker,a,b]);
assert.equal(hold.code,0,hold.stderr);
assert.equal(blocked,true,"two different Proposals must actually wait on one PostgreSQL lock");
for(const r of [resultA,resultB]) assert.equal(r.code,0,r.err||r.out);
const all=resultA.out+resultB.out;
assert.equal((all.match(/COMPETING_WINNER=/g)||[]).length,1);
assert.equal((all.match(/COMPETING_LOSER=/g)||[]).length,1);
const winner=/COMPETING_WINNER=([ab])/u.exec(all)?.[1];
const loser=winner==="a"?"b":"a";

async function state(){
  const r=await sql(`
SELECT json_build_object(
 'candidate_status',(SELECT status FROM public.learning_candidates WHERE id='${candidate}'),
 'revision',(SELECT revision FROM public.learning_candidates WHERE id='${candidate}'),
 'owner',(SELECT metadata->>'proposalId' FROM public.learning_candidates WHERE id='${candidate}'),
 'claims',(SELECT count(*) FROM public.jessica_candidate_skill_publications WHERE candidate_id='${candidate}'),
 'claim_owner',(SELECT proposal_id FROM public.jessica_candidate_skill_publications WHERE candidate_id='${candidate}'),
 'versions_a',(SELECT count(*) FROM public.jessica_experience_skill_versions WHERE skill_id='ci-competing-skill-a'),
 'versions_b',(SELECT count(*) FROM public.jessica_experience_skill_versions WHERE skill_id='ci-competing-skill-b'),
 'status_a',(SELECT status FROM public.learning_proposals WHERE id='ci-competing-proposal-a'),
 'status_b',(SELECT status FROM public.learning_proposals WHERE id='ci-competing-proposal-b'),
 'approved_a',(SELECT approved_at FROM public.learning_proposals WHERE id='ci-competing-proposal-a'),
 'approved_b',(SELECT approved_at FROM public.learning_proposals WHERE id='ci-competing-proposal-b')
)::text;
`);
  assert.equal(r.code,0,r.stderr);
  return JSON.parse(r.stdout);
}
const saved=await state();
assert.equal(saved.candidate_status,"PROMOTED");
assert.equal(saved.revision,1);
assert.equal(saved.claims,1);
assert.equal(saved.owner,"ci-competing-proposal-"+winner);
assert.equal(saved.claim_owner,saved.owner);
assert.equal(saved["versions_"+winner],1);
assert.equal(saved["versions_"+loser],0);
assert.equal(saved["status_"+winner],"APPROVED");
assert.equal(saved["status_"+loser],"FAILED");
assert.ok(saved["approved_"+winner]);
assert.equal(saved["approved_"+loser],null);

const replay=await worker(winner);
assert.equal(replay.code,0,replay.err||replay.out);
assert.match(replay.out,/COMPETING_WINNER=/);
assert.deepEqual(await state(),saved,"winner replay must not mutate any durable row");

// Eight independently restarted losing workers race after the terminal FAILED
// commit. The production FAILED status CAS must confirm the existing record
// and never create a new version, Candidate claim or timestamp.
const loserReplays=await Promise.all(Array.from({length:8},()=>worker(loser)));
for(const response of loserReplays){
  assert.equal(response.code,0,response.err||response.out);
  assert.match(response.out,new RegExp("COMPETING_LOSER_REPLAY="+loser));
}
assert.deepEqual(await state(),saved,
  "parallel losing Proposal replays must preserve the entire terminal state");

// The initial winner and loser must still be stable after all retries.
const finalWinner=await worker(winner);
const finalLoser=await worker(loser);
assert.equal(finalWinner.code,0,finalWinner.err||finalWinner.out);
assert.equal(finalLoser.code,0,finalLoser.err||finalLoser.out);
assert.match(finalWinner.out,/COMPETING_WINNER=/);
assert.match(finalLoser.out,/COMPETING_LOSER_REPLAY=/);
assert.deepEqual(await state(),saved,"final replay cannot steal Candidate ownership");
console.log("PASS: losing Proposal FAILED is stable across 10 safe worker replays");
