// Scope cancellation to a single AI invocation; never retry after abort.
export function throwIfAIAborted(signal){
  if(signal?.aborted){
    const error=new Error("AI request aborted");
    error.name="AbortError";
    error.code="ABORT_ERR";
    throw error;
  }
}

export function sleepAIWithAbort(milliseconds,signal){
  throwIfAIAborted(signal);
  if(!signal)return new Promise(resolve=>setTimeout(resolve,Math.max(0,milliseconds||0)));
  return new Promise((resolve,reject)=>{
    let timer;
    const cancel=()=>{
      clearTimeout(timer);
      signal.removeEventListener("abort",cancel);
      try{throwIfAIAborted(signal);}catch(error){reject(error);}
    };
    timer=setTimeout(()=>{
      signal.removeEventListener("abort",cancel);
      resolve();
    },Math.max(0,milliseconds||0));
    signal.addEventListener("abort",cancel,{once:true});
    if(signal.aborted)cancel();
  });
}
