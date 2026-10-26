import test from "node:test";
import assert from "node:assert/strict";
import {resolveStoredExperienceStatus}
 from "../experience/storage/experiencePersistence/experienceStoredStatus.js";

test("replayed draft Skill never masquerades as published",()=>{
 const replay={success:true,existing:true,status:"published",
   version:1,experience:{status:"draft",enabled:false}};
 assert.equal(resolveStoredExperienceStatus(replay),"draft");
});

test("replayed disabled Skill stays disabled",()=>{
 const replay={success:true,existing:true,status:"published",
   version:5,experience:{status:"disabled",enabled:false}};
 assert.equal(resolveStoredExperienceStatus(replay),"disabled");
});

test("unknown replay status fails closed instead of publishing",()=>{
 assert.equal(resolveStoredExperienceStatus(
   {existing:true,status:"published",experience:{}}),null);
});

test("new version status follows confirmed storage outcome",()=>{
 assert.equal(resolveStoredExperienceStatus(
   {existing:false,status:"published"}),"published");
 assert.equal(resolveStoredExperienceStatus(
   {existing:false,status:"draft"}),"draft");
});
