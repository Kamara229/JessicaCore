// Isolated async model of PostgreSQL Candidate row locks and immutable
// Proposal/History uniqueness. It never connects to production storage.
export function parallelSkillPublicationStore({preflightCount=0,ackLoss=false}={}){
  const receipts=new Map();
  const claims=new Map();
  const histories=new Map();
  const locks=new Map();
  let awaiting=0;
  let unblock;
  const gate=preflightCount>0?new Promise(resolve=>{unblock=resolve;}):null;
  const state={
    saveAttempts:0,commits:0,conflicts:0,lostAcks:0,
    inFlight:0,maxInFlight:0,readbacks:0,
    receipts,claims,histories
  };
  async function lookup(proposalId){
    state.readbacks++;
    if(gate&&awaiting++<preflightCount){
      if(awaiting===preflightCount)unblock();
      await gate;
    }
    const receipt=receipts.get(proposalId);
    return receipt?structuredClone(receipt):null;
  }
  async function save(experience){
    state.saveAttempts++;
    const proposalId=experience?.metadata?.proposalId;
    const candidateId=experience?.metadata?.candidateMemoryId;
    if(!proposalId||!candidateId)throw Error("Missing Candidate/Proposal");
    const earlier=locks.get(candidateId)||Promise.resolve();
    let release;
    const thisTurn=new Promise(resolve=>{release=resolve;});
    locks.set(candidateId,earlier.then(()=>thisTurn));
    await earlier;
    state.inFlight++;
    state.maxInFlight=Math.max(state.maxInFlight,state.inFlight);
    try {
      // Yield inside Candidate lock so concurrent executions really contend.
      await new Promise(resolve=>setTimeout(resolve,1));
      const previous=receipts.get(proposalId);
      if(previous){
        return {success:true,existing:true,
          skillId:previous.skillId,version:previous.version,
          experience:structuredClone(previous.payload)};
      }
      const owner=claims.get(candidateId);
      if(owner && owner.proposalId!==proposalId){
        state.conflicts++;
        throw Error("Candidate publication unique conflict (23505)");
      }
      const key=experience.id+":"+experience.version;
      if(histories.has(key))
        throw Error("Immutable Skill version unique conflict (23505)");
      const receipt={skillId:experience.id,version:experience.version,
        payload:structuredClone(experience)};
      histories.set(key,receipt);
      receipts.set(proposalId,receipt);
      claims.set(candidateId,{proposalId,skillId:experience.id});
      state.commits++;
      if(ackLoss){
        ackLoss=false;state.lostAcks++;
        throw Error("Skill transaction COMMIT succeeded; ACK lost");
      }
      return {success:true,existing:false,
        skillId:experience.id,version:experience.version,
        experience:structuredClone(experience)};
    }finally{
      state.inFlight--;
      release();
    }
  }
  return {state,lookup,save};
}
