import assert from "node:assert/strict";
import {assertIsolatedCi,sql} from "./fixtures/candidateRacePgClient.mjs";

assertIsolatedCi();
const phase=process.argv[2];
const candidate="35b89f96-cfbf-45ab-943f-722e6b457487";
const proposal="ci-commit-gap-proposal";
const skill="ci-commit-gap-skill";
if(phase==="seed"){
  const r=await sql(`
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
  assert.equal(r.code,0,r.stderr);
}
if(!["seed","pending","final"].includes(phase))throw Error("Unexpected phase");
const r=await sql(`
SELECT
 (SELECT count(*) FROM public.jessica_experience_skill_versions
  WHERE skill_id='${skill}'),
 (SELECT count(*) FROM public.jessica_candidate_skill_publications
  WHERE candidate_id='${candidate}' AND proposal_id='${proposal}'),
 (SELECT status FROM public.learning_candidates WHERE id='${candidate}'),
 (SELECT revision FROM public.learning_candidates WHERE id='${candidate}'),
 (SELECT status FROM public.learning_proposals WHERE id='${proposal}'),
 (SELECT CASE WHEN approved_at IS NULL THEN 0 ELSE 1 END
  FROM public.learning_proposals WHERE id='${proposal}');
`);
assert.equal(r.code,0,r.stderr);
assert.equal(r.stdout,phase==="final"?"1|1|PROMOTED|1|APPROVED|1":
  "1|1|ACTIVE|0|PROCESSING|0","durable state across restart");
console.log("PASS: crash gap persisted state "+phase);
