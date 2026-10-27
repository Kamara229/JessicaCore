import assert from "node:assert/strict";
import {assertIsolatedCi,sql} from "./fixtures/candidateRacePgClient.mjs";

assertIsolatedCi();
const phase=process.argv[2];
const candidate="fcdf796b-1f12-4330-9be2-5c642dfbf356";
const proposal="ci-restarted-proposal";
const skill="ci-restarted-skill";

if(phase==="seed"){
  const r=await sql(`
BEGIN;
INSERT INTO public.learning_candidates(id,status)
 VALUES ('${candidate}','ACTIVE');
INSERT INTO public.learning_proposals(id,action,analysis,status,processing_at)
 VALUES ('${proposal}','NEW_SKILL',
 jsonb_build_object('candidateMemory',jsonb_build_object('id','${candidate}')),
 'PROCESSING',now());
INSERT INTO public.jessica_experience_skills(id) VALUES ('${skill}');
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
 VALUES ('${skill}',1,jsonb_build_object(
 'id','${skill}','version',1,'name','Committed restarted Skill',
 'metadata',jsonb_build_object('proposalId','${proposal}')));
UPDATE public.learning_candidates SET status='PROMOTED',
 metadata=jsonb_build_object('proposalId','${proposal}','skillId','${skill}')
 WHERE id='${candidate}';
COMMIT;
`);
  assert.equal(r.code,0,r.stderr);
}else if(phase==="verify"){
  const r=await sql(`
SELECT
 (SELECT count(*) FROM public.jessica_experience_skill_versions
  WHERE skill_id='${skill}'),
 (SELECT count(*) FROM public.jessica_candidate_skill_publications
  WHERE candidate_id='${candidate}' AND proposal_id='${proposal}'
  AND skill_id='${skill}' AND skill_version=1),
 (SELECT count(*) FROM public.learning_candidates
  WHERE id='${candidate}' AND status='PROMOTED'
  AND metadata->>'proposalId'='${proposal}'),
 (SELECT count(*) FROM public.learning_proposals
  WHERE id='${proposal}' AND status='APPROVED'
  AND approved_at IS NOT NULL AND processing_at IS NULL);
`);
  assert.equal(r.code,0,r.stderr);
  assert.equal(r.stdout,"1|1|1|1","exactly one Skill, Candidate, Claim, approved Proposal");
}else throw Error("Expected seed or verify");
console.log("PASS: real persisted Proposal restart "+phase);
