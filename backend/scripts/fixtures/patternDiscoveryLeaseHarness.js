import {randomUUID} from "node:crypto";

// Async Pattern Discovery fixture: the caller controls when AI extraction ends.
export function patternDiscoveryLeaseHarness(){
  const row={id:"pattern-lease-item",status:"PROCESSING",
    claim_token:randomUUID()};
  let startedResolve,finishResolve;
  const started=new Promise(resolve=>{startedResolve=resolve;});
  const finish=new Promise(resolve=>{finishResolve=resolve;});
  const counts={checks:0,memory:0,proposals:0,updates:0,retries:0};
  const client={async rpc(name,args){
    if(name!=="is_learning_queue_claim_current")
      throw new Error("Unexpected RPC "+name);
    counts.checks++;
    return {data:row.id===args.p_queue_item_id
      &&row.status==="PROCESSING"
      &&row.claim_token===args.p_claim_token,error:null};
  }};
  const prepareDiscoveryStep=async item=>{
    startedResolve();
    const result=await finish;
    return result||{
      success:true,ignored:false,resolved:true,
      queueItem:{
        ...item,id:"fabricated-from-AI",action:"NEW_SKILL",
        event:{payload:{skillCandidate:{name:"Transient pattern"}}}
      },
      discovery:{source:"mock-pattern-extractor"}
    };
  };
  return {
    row,client,counts,started,
    prepareDiscoveryStep,
    complete(result=null){finishResolve(result);},
    expire(){row.status="FAILED";row.claim_token=null;},
    rotate(){row.status="PROCESSING";row.claim_token=randomUUID();},
    get item(){return {
      id:row.id,action:"PATTERN_DISCOVERY",
      claimToken:"",event:{payload:{discovery:{}}}
    };},
  };
}
