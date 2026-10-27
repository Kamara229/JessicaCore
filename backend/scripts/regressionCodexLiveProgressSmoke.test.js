import test from "node:test";
import assert from "node:assert/strict";
import {interpretRelayTurn}
  from "../core/codex/relayInterpreter.js";
import {verifyCurrentBackendProgress}
  from "./smokeCodexMainProgress.mjs";

test("real Codex progress scenario uses no interpreter quota or test chat",async()=>{
  let providerCalls=0;
  let requests=0;
  await verifyCurrentBackendProgress(async(payload)=>{
    requests++;
    return interpretRelayTurn(payload,async()=>{
      providerCalls++;
      throw Error("quota exceeded");
    });
  });
  assert.equal(requests,1);
  assert.equal(providerCalls,0);
});
