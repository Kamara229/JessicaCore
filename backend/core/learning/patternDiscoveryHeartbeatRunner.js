import {renewLearningQueueClaim} from "./learningQueueHeartbeat.js";

const INTERVAL_MS=30000;
const TIMEOUT_MS=300000;

// Supervise only long-running discovery, not whole Worker. Promise.race
// ends processing immediately after ownership loss even when AI ignores abort.
export async function runPatternDiscoveryWithHeartbeat({
  item,work,heartbeat=renewLearningQueueClaim,
  intervalMs=INTERVAL_MS,timeoutMs=TIMEOUT_MS
}={}){
  const token=item?.claimToken||item?.claim_token||null;
  if(!item?.id||!token||typeof work!=="function")
    return {ok:false,stale:true,reason:"Pattern Discovery claim is missing"};
  const interval=Math.max(5,Number.isFinite(intervalMs)?intervalMs:INTERVAL_MS);
  const deadline=Math.max(interval+1,
    Number.isFinite(timeoutMs)?timeoutMs:TIMEOUT_MS);
  const controller=new AbortController();
  let finished=false,beatTimer=null,deadlineTimer=null,notifyFailure;
  const failure=new Promise(resolve=>{notifyFailure=resolve;});
  function stop(outcome){
    if(finished)return;
    finished=true;
    controller.abort();
    notifyFailure(outcome);
  }
  function schedule(){
    if(finished)return;
    beatTimer=setTimeout(()=>{void beat();},interval);
  }
  async function beat(){
    if(finished)return;
    let result;
    try{
      result=await heartbeat({queueItemId:item.id,claimToken:token});
    }catch(error){
      result={renewed:false,stale:false,reason:error?.message||"Heartbeat error"};
    }
    if(finished)return;
    if(result?.renewed!==true){
      stop({ok:false,stale:result?.stale===true,
        reason:result?.reason||"Pattern Discovery heartbeat unverified"});
    }else schedule();
  }
  schedule();
  deadlineTimer=setTimeout(()=>stop({
    ok:false,stale:false,timedOut:true,
    reason:"Pattern Discovery timed out before verified completion"
  }),deadline);
  try{
    return await Promise.race([
      Promise.resolve().then(()=>work({signal:controller.signal}))
        .then(value=>({ok:true,value}),
          error=>({ok:false,stale:false,reason:error?.message||"Discovery failed"})),
      failure
    ]);
  }finally{
    finished=true;
    clearTimeout(beatTimer);
    clearTimeout(deadlineTimer);
    controller.abort();
  }
}
