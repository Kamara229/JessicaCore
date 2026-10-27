import assert from "node:assert/strict";
import {persistExperienceSkill}
  from "../../backend/experience/storage/experiencePersistence/experienceWriter.js";
import {assertIsolatedCi,sql} from "./fixtures/candidateRacePgClient.mjs";

// The database is disposable: isolated Postgres from GitHub Actions.
assertIsolatedCi();
const candidate="d52e760f-c0a1-4dae-8b3f-a374ffbcd430";
const proposalId="ci-real-ack-proposal";
const skillId="ci-real-ack-skill";
const setup=await sql(`
INSERT INTO public.learning_candidates(id,status) VALUES('${candidate}','ACTIVE');
INSERT INTO public.learning_proposals(id,action,analysis) VALUES(
 '${proposalId}','NEW_SKILL',
 jsonb_build_object('candidateMemory',jsonb_build_object('id','${candidate}')));
INSERT INTO public.jessica_experience_skills(id) VALUES('${skillId}');
`);
assert.equal(setup.code,0,setup.stderr);

const experience={
  id:skillId,name:"Real committed Skill ACK loss probe",version:1,
  previousVersion:null,mode:"create",enabled:false,
  workflow:["Read back immutable Skill receipt after lost ACK"],
  metadata:{proposalId,candidateMemoryId:candidate}
};
let writeAttempts=0;
let lookupCount=0;
const lookup=async id=>{
  assert.equal(id,proposalId);
  lookupCount++;
  const found=await sql(`
SELECT json_build_object('skillId',skill_id,'version',version,'payload',payload)::text
FROM public.jessica_experience_skill_versions
WHERE payload #>> '{metadata,proposalId}'='${proposalId}'
ORDER BY version DESC LIMIT 1;
`);
  assert.equal(found.code,0,found.stderr);
  return found.stdout?JSON.parse(found.stdout):null;
};
const save=async value=>{
  writeAttempts++;
  const payload=JSON.stringify(value).replaceAll("'","''");
  const committed=await sql(`
BEGIN;
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES ('${skillId}',1,'${payload}'::jsonb);
COMMIT;
`);
  assert.equal(committed.code,0,committed.stderr);
  // PostgreSQL really committed; the application intentionally loses its ACK.
  throw Error("simulated network ACK lost after PostgreSQL COMMIT");
};
const saved=await persistExperienceSkill(experience,{lookup,save});
assert.equal(saved.success,true);
assert.equal(saved.existing,true);
assert.equal(saved.commitRecovered,true);
assert.equal(saved.skillId,skillId);
assert.equal(saved.version,1);
assert.equal(writeAttempts,1);
assert.ok(lookupCount>=2,"must read durable receipt after lost ACK");

// Every replay must use the durable receipt; never write another version.
const replays=await Promise.all(Array.from({length:24},()=>
  persistExperienceSkill(experience,{lookup,save})));
assert.equal(replays.length,24);
assert.ok(replays.every(v=>v.success===true && v.existing===true
  && v.skillId===skillId && v.version===1));
assert.equal(writeAttempts,1,"recovery and replays cannot retry the write");

const counts=await sql(`
SELECT
 (SELECT count(*) FROM public.jessica_experience_skill_versions
  WHERE skill_id='${skillId}'),
 (SELECT count(*) FROM public.jessica_candidate_skill_publications
  WHERE candidate_id='${candidate}' AND proposal_id='${proposalId}'
  AND skill_id='${skillId}' AND skill_version=1);
`);
assert.equal(counts.code,0,counts.stderr);
assert.equal(counts.stdout,"1|1","no duplicate history or Candidate claim");
console.log("PASS: real COMMIT, simulated lost ACK, durable receipt, 24 safe replays");
