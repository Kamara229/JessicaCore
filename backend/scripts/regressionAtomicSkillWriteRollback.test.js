import test from "node:test";
import assert from "node:assert/strict";
import {persistExperienceSkill}
  from "../experience/storage/experiencePersistence/experienceWriter.js";

const candidate="candidate-atomic-write-regression";
const skill="atomic-write-skill";
const proposal="atomic-write-proposal";
const experience={
  id:skill,name:"Atomic Skill publication regression",
  version:1,previousVersion:null,mode:"create",
  metadata:{proposalId:proposal,candidateMemoryId:candidate},
  workflow:["Isolated regression of atomic transaction"],
  examples:[{task:"rollback verification",success:true}]
};

function atomicStore(){
  // The real v4 semantics are tested independently with PostgreSQL SQL
  // rollback-only fixture. This fake tests caller behaviour on its responses.
  const state={skill:null,history:[],modules:[],evidence:[],claim:null,
    audits:[],attempts:0,lateModuleFailure:true};
  return {
    state,
    lookup:async id=>{
      assert.equal(id,proposal);
      const version=state.history.find(v=>v.payload.metadata.proposalId===id);
      return version?structuredClone(version):null;
    },
    save:async value=>{
      state.attempts++;
      if(state.lateModuleFailure){
        state.lateModuleFailure=false;
        // v4 transaction rolls back evidence, current, History,
        // Candidate claim, audit and the first inserted module.
        throw Error("duplicate Skill module part, SQLSTATE 23505");
      }
      if(state.claim&&state.claim!==value.metadata.proposalId)
        throw Error("Candidate already claimed, SQLSTATE 23505");
      if(state.history.length)
        return {success:true,existing:true,skillId:skill,version:1};
      state.skill={id:skill,version:1};
      state.history.push({skillId:skill,version:1,
        payload:structuredClone(value)});
      state.modules.push({skillId:skill,version:1,part:0});
      state.evidence.push({skillId:skill,trace:"atomic-trace"});
      state.claim=value.metadata.proposalId;
      state.audits.push({skillId:skill,version:1});
      return {success:true,existing:false,skillId:skill,version:1,
        experience:structuredClone(value)};
    }
  };
}

test("late v4 failure defers Approval without leaking partial Skill transaction",async()=>{
  const db=atomicStore();
  const first=await persistExperienceSkill(experience,db);
  assert.equal(first.success,false);
  assert.equal(first.uncertain,true);
  assert.equal(db.state.attempts,1);
  for(const table of ["history","modules","evidence","audits"])
    assert.equal(db.state[table].length,0,table);
  assert.equal(db.state.skill,null);
  assert.equal(db.state.claim,null);

  const recovered=await persistExperienceSkill(experience,db);
  assert.equal(recovered.success,true);
  assert.equal(recovered.existing,false);
  assert.equal(recovered.version,1);
  assert.equal(db.state.history.length,1);
  assert.equal(db.state.modules.length,1);
  assert.equal(db.state.evidence.length,1);
  assert.equal(db.state.audits.length,1);
  assert.equal(db.state.claim,proposal);

  const duplicate=await persistExperienceSkill(experience,db);
  assert.equal(duplicate.success,true);
  assert.equal(duplicate.existing,true);
  assert.equal(duplicate.version,1);
  assert.equal(db.state.attempts,2,
    "Idempotent replay must use immutable receipt, not RPC");
  for(const table of ["history","modules","evidence","audits"])
    assert.equal(db.state[table].length,1,table);
});

test("concurrent replays after durable publish do not invoke a second write",async()=>{
  const db=atomicStore();db.state.lateModuleFailure=false;
  const committed=await persistExperienceSkill(experience,db);
  assert.equal(committed.success,true);
  const results=await Promise.all(Array.from({length:12},
    ()=>persistExperienceSkill(experience,db)));
  assert.ok(results.every(result=>result.success&&result.existing));
  assert.equal(db.state.attempts,1);
  assert.equal(db.state.history.length,1);
  assert.equal(db.state.modules.length,1);
  assert.equal(db.state.evidence.length,1);
  assert.equal(db.state.claim,proposal);
});
