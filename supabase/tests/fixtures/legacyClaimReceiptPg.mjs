import assert from "node:assert/strict";
import {assertIsolatedCi,sql} from "./candidateRacePgClient.mjs";

// Real PostgreSQL receipt readback; a wrong local port simulates a lost DB
// connection while the committed row remains intact in the CI database.
export function legacyClaimReceiptLookup(proposalId){
  assertIsolatedCi();
  let outage=false,calls=0;
  return {
    get calls(){return calls;},
    setOutage(value){outage=value===true;},
    async lookup(id){
      assert.equal(id,proposalId);
      calls++;
      const result=await sql(`
SELECT json_build_object('skillId',skill_id,'version',version,'payload',payload)::text
FROM public.jessica_experience_skill_versions
WHERE payload #>> '{metadata,proposalId}'='${proposalId}'
ORDER BY version DESC LIMIT 1;
`,"jessica-ci-legacy-ack-receipt",outage?{PGPORT:"1"}:{});
      if(result.code!==0)
        throw Error("Legacy claim receipt readback unavailable: "+result.stderr);
      return result.stdout?JSON.parse(result.stdout):null;
    }
  };
}
