import assert from "node:assert/strict";
import {assertIsolatedCi,sql,waitFor} from "./candidateRacePgClient.mjs";

// The real COMMIT ACK is deliberately lost by the caller, but two
// independent transactions must already have waited on its Candidate lock.
export async function raceLegacyClaimDuringCommit({
  candidate,owner,skill,loser,loserSkill,payload
}){
  assertIsolatedCi();
  const body=JSON.stringify(payload).replaceAll("'","''");
  const held=sql(`
BEGIN;
SELECT id FROM public.learning_candidates WHERE id='${candidate}' FOR UPDATE;
SELECT pg_sleep(7);
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES('${skill}',2,'${body}'::jsonb);
COMMIT;
`,"jessica-ci-legacy-ack-holder");
  assert.equal(await waitFor(`
SELECT CASE WHEN EXISTS(SELECT 1 FROM pg_stat_activity
 WHERE application_name='jessica-ci-legacy-ack-holder'
 AND state='active' AND query LIKE '%pg_sleep(7)%') THEN 1 ELSE 0 END;
`,4500),true,"original publisher must hold the Candidate row lock");

  const replay=sql(`
BEGIN;
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES('${skill}',3,jsonb_build_object('id','${skill}',
 'metadata',jsonb_build_object('proposalId','${owner}')));
COMMIT;
`,"jessica-ci-legacy-ack-same-owner");
  const theft=sql(`
BEGIN;
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES('${loserSkill}',1,jsonb_build_object('id','${loserSkill}',
 'metadata',jsonb_build_object('proposalId','${loser}')));
COMMIT;
`,"jessica-ci-legacy-ack-other-owner");
  const bothBlocked=await waitFor(`
SELECT CASE WHEN count(*)=2 THEN 1 ELSE 0 END
FROM pg_stat_activity WHERE application_name IN
 ('jessica-ci-legacy-ack-same-owner','jessica-ci-legacy-ack-other-owner')
 AND wait_event_type='Lock';
`,5500);
  const [committed,replayed,stolen]=await Promise.all([held,replay,theft]);
  assert.equal(bothBlocked,true,"two transactions must wait on PostgreSQL locks");
  assert.equal(committed.code,0,committed.stderr);
  assert.notEqual(replayed.code,0,"owner cannot publish second immutable claim version");
  assert.match(replayed.stderr,/different Skill or Proposal already claimed Candidate/);
  assert.notEqual(stolen.code,0,"loser cannot steal legacy owner");
  assert.match(stolen.stderr,/different Proposal already owns Candidate/);
}
