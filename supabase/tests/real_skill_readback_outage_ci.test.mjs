import assert from "node:assert/strict";
import {persistExperienceSkill}
  from "../../backend/experience/storage/experiencePersistence/experienceWriter.js";
import {assertIsolatedCi,sql} from "./fixtures/candidateRacePgClient.mjs";

assertIsolatedCi();
const candidateId="59bc3871-58ba-4bc8-a82f-baa1bc504610";
const proposalId="ci-readback-outage-proposal";
const skillId="ci-readback-outage-skill";

const init=await sql(`
INSERT INTO public.learning_candidates(id,status)
VALUES ('${candidateId}','ACTIVE');
INSERT INTO public.learning_proposals(id,action,analysis)
VALUES ('${proposalId}','NEW_SKILL',
 jsonb_build_object('candidateMemory',jsonb_build_object('id','${candidateId}')));
INSERT INTO public.jessica_experience_skills(id) VALUES ('${skillId}');
`);
assert.equal(init.code,0,init.stderr);

const experience={
  id:skillId,name:"Uncertain Skill receipt during PostgreSQL readback outage",
  version:1,previousVersion:null,mode:"create",enabled:false,
  workflow:["Do not retry an unverified committed Skill"],
  metadata:{proposalId,candidateMemoryId:candidateId}
};
let writes=0;
let outage=false;
const lookup=async id=>{
  assert.equal(id,proposalId);
  const r=await sql(`
SELECT json_build_object('skillId',skill_id,'version',version,'payload',payload)::text
FROM public.jessica_experience_skill_versions
WHERE payload #>> '{metadata,proposalId}'='${proposalId}'
ORDER BY version DESC LIMIT 1;
`,"jessica-ci-readback",outage?{PGPORT:"1"}:{});
  if(r.code!==0)throw Error("PostgreSQL receipt readback unavailable: "+r.stderr);
  return r.stdout?JSON.parse(r.stdout):null;
};
const save=async value=>{
  writes++;
  const payload=JSON.stringify(value).replaceAll("'","''");
  const r=await sql(`
BEGIN;
INSERT INTO public.jessica_experience_skill_versions(skill_id,version,payload)
VALUES('${skillId}',1,'${payload}'::jsonb);
COMMIT;
`);
  assert.equal(r.code,0,r.stderr);
  outage=true; // After real COMMIT, a different readback connection goes offline.
  throw Error("Save acknowledgement lost after COMMIT");
};

// Readable preflight -> durable COMMIT -> lost ACK + unavailable readback.
const initial=await persistExperienceSkill(experience,{lookup,save});
assert.equal(initial.success,false);
assert.equal(initial.existing,false);
assert.equal(initial.uncertain,true);
assert.match(initial.error,/readback unavailable/);
assert.equal(writes,1);

// Independent healthy connection verifies the Skill was actually committed.
const state=await sql(`
SELECT (SELECT count(*) FROM public.jessica_experience_skill_versions
 WHERE skill_id='${skillId}'),
 (SELECT count(*) FROM public.jessica_candidate_skill_publications
 WHERE candidate_id='${candidateId}' AND proposal_id='${proposalId}');
`);
assert.equal(state.code,0,state.stderr);
assert.equal(state.stdout,"1|1","one immutable History and publication claim");

// Replays must fail closed while preflight cannot confirm the receipt.
const deferred=await Promise.all(Array.from({length:8},()=>
  persistExperienceSkill(experience,{lookup,save})));
assert.ok(deferred.every(r=>r.success===false && r.uncertain===true));
assert.equal(writes,1,"no second write during uncertain readback");

// Restored readback confirms original saved receipt without another COMMIT.
outage=false;
const resumed=await persistExperienceSkill(experience,{lookup,save});
assert.equal(resumed.success,true);
assert.equal(resumed.existing,true);
assert.equal(resumed.skillId,skillId);
assert.equal(resumed.version,1);
assert.equal(writes,1,"recovery must not write or create a new Skill version");
console.log("PASS: COMMIT + ACK loss + readback outage -> uncertain, no duplicate writes");
