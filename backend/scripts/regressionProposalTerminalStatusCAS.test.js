import test from "node:test";
import assert from "node:assert/strict";
import {transitionProposalStatus}
  from "../core/learning/proposalStorage/proposalStatusCAS.js";

const id="00000000-0000-4000-8000-000000000831";
function store(initialStatus="PROCESSING"){
  let row={id,status:initialStatus,processing_at:"2026-10-27T17:00:00Z",
    approved_at:null,rejected_at:null};
  let writes=0,lostAck=false,readError=false;
  const client={from(table){
    assert.equal(table,"learning_proposals");
    const q={payload:null,filters:{},
      update(payload){this.payload=payload;return this;},
      select(){return this;},
      eq(key,value){this.filters[key]=value;return this;},
      async maybeSingle(){
        if(this.payload===null){
          if(readError)return {data:null,error:{message:"readback offline"}};
          const found=row&&Object.entries(this.filters)
            .every(([k,v])=>row[k]===v);
          return {data:found?structuredClone(row):null,error:null};
        }
        const found=row&&Object.entries(this.filters)
          .every(([k,v])=>row[k]===v);
        if(!found)return {data:null,error:null};
        row={...row,...this.payload};
        writes++;
        if(lostAck){lostAck=false;
          return {data:null,error:{message:"status ACK lost after COMMIT"}};
        }
        return {data:structuredClone(row),error:null};
      }
    };
    return q;
  }};
  return {client,get row(){return structuredClone(row);},
    get writes(){return writes;},
    loseNextAck(){lostAck=true;},
    readbackUnavailable(){readError=true;}}
}
function status(db,value){
  const payload={status:value,processing_at:null,
    approved_at:value==="APPROVED"?"2026-10-27T18:00:00Z":null,
    rejected_at:value==="REJECTED"?"2026-10-27T18:00:00Z":null};
  return transitionProposalStatus({client:db.client,id,status:value,payload});
}

test("APPROVED Proposal stays immutable against stale FAILED or REJECTED Worker",async()=>{
  const db=store();
  const approved=await status(db,"APPROVED");
  assert.equal(approved.success,true);
  const before=db.row;
  for(const next of ["FAILED","REJECTED","KEEP_CANDIDATE",
    "PENDING_APPROVAL","PROCESSING"]){
    const failed=await status(db,next);
    assert.equal(failed.success,false);
    assert.equal(failed.conflict,true);
    assert.equal(db.writes,1);
    assert.deepEqual(db.row,before);
  }
});

test("repeat APPROVED confirmation does not change timestamp or write twice",async()=>{
  const db=store();
  await status(db,"APPROVED");
  const original=db.row;
  const second=await status(db,"APPROVED");
  assert.equal(second.success,true);
  assert.equal(second.existing,true);
  assert.equal(second.updated,false);
  assert.equal(db.writes,1);
  assert.deepEqual(db.row,original);
});

test("lost ACK after APPROVED commit is recovered from durable readback",async()=>{
  const db=store();
  db.loseNextAck();
  const result=await status(db,"APPROVED");
  assert.equal(result.success,true);
  assert.equal(result.existing,true);
  assert.equal(result.recoveredAck,true);
  assert.equal(db.row.status,"APPROVED");
  assert.equal(db.writes,1);
});

test("lost ACK without readback remains uncertain and cannot claim approval",async()=>{
  const db=store();
  db.loseNextAck();
  db.readbackUnavailable();
  const result=await status(db,"APPROVED");
  assert.equal(result.success,false);
  assert.equal(result.uncertain,true);
  assert.match(result.error,/readback offline/);
  assert.equal(db.writes,1);
  assert.equal(db.row.status,"APPROVED");
});

test("parallel conflicting finalizers allow exactly one terminal status",async()=>{
  const db=store();
  const [a,b]=await Promise.all([status(db,"APPROVED"),status(db,"REJECTED")]);
  assert.equal([a,b].filter(x=>x.success).length,1);
  assert.equal([a,b].filter(x=>x.conflict).length,1);
  assert.equal(db.writes,1);
  assert.ok(["APPROVED","REJECTED"].includes(db.row.status));
});

test("PROCESSING requires a pending Proposal and cannot reopen finalization",async()=>{
  const db=store("PENDING_APPROVAL");
  const claim=await status(db,"PROCESSING");
  assert.equal(claim.success,true);
  assert.equal(db.row.status,"PROCESSING");
  const approved=await status(db,"APPROVED");
  assert.equal(approved.success,true);
  const reopened=await status(db,"PROCESSING");
  assert.equal(reopened.success,false);
  assert.equal(db.writes,2);
});
