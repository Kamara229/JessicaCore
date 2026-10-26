import test from "node:test";
import assert from "node:assert/strict";
import {planAuthenticatedStageClaimPage as page} from "../core/files/documentStageClaimPage.js";
import {makeStageClaimFixture,task,secret,source} from "./fixtures/documentStageClaimFixture.js";
test("stage claim pages reject absent or unsigned evidence",()=>{
 const {verified,reports}=makeStageClaimFixture();
 const untrusted=structuredClone(reports);
 delete untrusted[0].stageReport.authTag;
 assert.equal(page({source,verified,reports:untrusted,task,secret}),null);
 assert.equal(page({source,verified,reports:reports.slice(0,-1),task,secret}),null);
 assert.equal(page({source,verified,reports,task:"Другая задача",secret}),null);
 assert.equal(page({source,verified,reports,task,secret,claimOffset:-1}),null);
 const forged=structuredClone(reports);
 forged[0].text+=" изменение";
 assert.equal(page({source,verified,reports:forged,task,secret}),null);
});
test("complete verified stage reports page every claim without omissions",()=>{
 const {verified,reports}=makeStageClaimFixture();
 const claims=[];
 let offset=0,finished=false;
 for(let i=0;i<20;i++){
  const p=page({source,verified,reports,task,secret,claimOffset:offset});
  assert.ok(p);
  assert.equal(p.semanticSynthesisPerformed,false);
  assert.equal(p.reportOriginAuthenticated,true);
  assert.ok(p.returnedClaims>0&&p.returnedClaims<=12);
  claims.push(...p.claims);
  if(p.nextClaimOffset===null){finished=true;break;}
  assert.equal(p.nextClaimOffset,claims.length);
  offset=p.nextClaimOffset;
 }
 assert.equal(finished,true);
 assert.equal(claims.length,verified.totalWindows);
 assert.deepEqual(claims.map(c=>c.windowNumber),
  Array.from({length:verified.totalWindows},(_,i)=>i+1));
});
