import {randomUUID} from "node:crypto";
import {packageExperienceModules} from "../../experience/storage/skillModules/experienceModulePacker.js";

// Run two HTTP RPC calls using two separately constructed clients.
// This tests overlapping client requests, not proof of Postgres execution overlap.
export async function issueExperienceV4Race(createClient) {
  const sid="jessica-rpc-race-probe-"+randomUUID();
  const proposalId="jessica-rpc-race-proposal-"+randomUUID();
  const {facade,modules}=packageExperienceModules({
    id:sid,name:"Disposable RPC concurrency probe",version:1,
    previousVersion:null,enabled:false,
    workflow:["read source","verify output"]
  });
  const args={p_skill_id:sid,p_version:1,p_previous_version:null,
    p_mode:"create",p_enabled:false,p_payload:facade,
    p_metadata:{proposalId,learning:{autonomy:"TEST_ONLY"}},
    p_evidence:[],p_modules:modules};
  const clients=[createClient(),createClient()];
  if(!clients[0]||!clients[1]||clients[0]===clients[1])
    throw Error("Race test requires two independent Supabase clients");
  const starts=[];
  const requests=clients.map(async(client,index)=>{
    starts[index]=performance.now();
    const {data,error}=await client.rpc("save_jessica_experience_skill_v4",args);
    if(error)throw Error("RPC race call "+index+": "+error.message);
    return data;
  });
  const outcomes=await Promise.allSettled(requests);
  return {sid,proposalId,clients,outcomes,
    clientStartSkewMs:Math.abs(starts[0]-starts[1])};
}
