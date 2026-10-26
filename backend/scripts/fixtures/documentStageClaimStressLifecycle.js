import {Worker} from "node:worker_threads";

const url=new URL("./documentStageClaimConcurrencyWorker.js",import.meta.url);

/** Synthetic request lifecycle; never touches production provider or queues. */
export function runSyntheticDocumentRequest(id,{
    inject429=false,cancelOnStart=false
}={}){
    if(!Number.isSafeInteger(id)||id<1||id>10||
       (inject429&&cancelOnStart))
        return Promise.reject(Error("Invalid synthetic request"));
    return new Promise((resolve,reject)=>{
        const worker=new Worker(url,{workerData:{id,inject429},
            resourceLimits:{maxOldGenerationSizeMb:160}});
        let started=false,response=null,invalid=false;
        worker.on("message",msg=>{
            if(msg?.id!==id||msg?.phase==="started"&&started||
               msg?.phase!=="started"&&response!==null){
                invalid=true;
                worker.terminate().catch(reject);
                return;
            }
            if(msg.phase==="started"){
                started=true;
                if(cancelOnStart)worker.terminate().catch(reject);
            }else if(msg.phase==="complete"||msg.phase==="rate-limited"){
                response=msg;
            }else{
                invalid=true;
                worker.terminate().catch(reject);
            }
        });
        worker.once("error",reject);
        worker.once("exit",code=>{
            if(invalid||!started)return reject(
                Error("Invalid worker lifecycle "+id));
            if(cancelOnStart){
                if(response!==null||code===0)return reject(
                    Error("Cancelled request yielded an output "+id));
                return resolve({phase:"cancelled",id,noVerifiedResult:true});
            }
            if(code!==0||!response)return reject(
                Error("Worker missing result "+id+" exit "+code));
            if(inject429&&
               (response.phase!=="rate-limited"||
                response.status!==429||response.blocked!==true||
                response.noVerifiedResult!==true))
                return reject(Error("Missing 429 fast-fail state "+id));
            if(!inject429&&response.phase!=="complete")
                return reject(Error("Unexpected incomplete result "+id));
            resolve(response);
        });
    });
}
