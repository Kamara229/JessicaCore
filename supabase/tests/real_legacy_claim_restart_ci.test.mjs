import assert from "node:assert/strict";
import {execFile} from "node:child_process";
import {promisify} from "node:util";
import {assertIsolatedCi,sql} from "./fixtures/candidateRacePgClient.mjs";

assertIsolatedCi();
const exec=promisify(execFile);
const candidate="ea103772-d67b-4a0d-a6e7-8cc0bb3f849b";
const skill="ci-legacy-ack-skill",loserSkill="ci-legacy-ack-loser-skill";
const owner="ci-legacy-ack-owner",loser="ci-legacy-ack-loser";
const snapshot=async()=>{
  const r=await sql(`
SELECT
 (SELECT count(*) FROM public.jessica_candidate_skill_publications
 WHERE candidate_id='${candidate}'),
 (SELECT proposal_id FROM public.jessica_candidate_skill_publications
 WHERE candidate_id='${candidate}'),
 (SELECT skill_version FROM public.jessica_candidate_skill_publications
 WHERE candidate_id='${candidate}'),
 (SELECT count(*) FROM public.jessica_experience_skill_versions
 WHERE skill_id='${skill}'),
 (SELECT count(*) FROM public.jessica_experience_skill_versions
 WHERE skill_id='${loserSkill}'),
 (SELECT metadata->>'proposalId' FROM public.learning_candidates
 WHERE id='${candidate}'),
 (SELECT status FROM public.learning_proposals WHERE id='${loser}');
`);
  assert.equal(r.code,0,r.stderr);
  return r.stdout;
};
const state=await snapshot();
// Previous CI step committed the claim even when its acknowledgement
// and subsequent PostgreSQL receipt readback were both lost.
assert.equal(state,`1|${owner}|2|2|0|${owner}|PROCESSING`);

const worker=async mode=>{
  const {stdout,stderr}=await exec(process.execPath,[
    new URL("./fixtures/legacyAckRestartWorker.mjs",import.meta.url).pathname,
    mode
  ],{env:process.env,timeout:15000});
  assert.match(stdout,new RegExp("PASS legacy receipt restart "+mode));
  assert.equal(stderr.trim(),"");
};
for(let i=0;i<8;i++)
  await worker("offline");
await Promise.all(Array.from({length:4},()=>worker("corrupt")));
for(let i=0;i<4;i++)
  await Promise.all(Array.from({length:4},()=>worker("restored")));
assert.equal(await snapshot(),state,
  "eight offline, four corrupt and sixteen restored subprocesses cannot change durable owner/history");
console.log("PASS: legacy claim survived 28 independent worker restarts, no duplicate writes");
