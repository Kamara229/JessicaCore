import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {codexConfiguration,startCodexDay} from "../core/codex/codexSessionService.js";

test("Day mode never needs a paid AI API key",()=>{
    const config=codexConfiguration();
    assert.equal(config.apiFreeDayMode,true);
    const flow=readFileSync(new URL("../../.github/workflows/jessica-day.yml",import.meta.url),"utf8");
    assert.doesNotMatch(flow,/OPENAI_API_KEY|GROQ_API_KEY|codex-action/);
    assert.match(flow,/workflow_dispatch/);
    assert.match(flow,/persist-credentials: false/);
    assert.match(flow,/gh pr create/);
    assert.doesNotMatch(flow,/gh pr merge/);
});
test("Owner started day run dispatches only fixed objective with isolated issue",async()=>{
    const oldOwner=process.env.JESSICA_CODEX_OWNER_KEY;
    const oldGithub=process.env.JESSICA_GITHUB_AUTOMATION_TOKEN;
    try{
        process.env.JESSICA_CODEX_OWNER_KEY="test-owner-key-very-long-12345";
        process.env.JESSICA_GITHUB_AUTOMATION_TOKEN="test-token";
        const calls=[];
        const api=async(path,args={})=>{
            calls.push({path,args});
            if(path.includes("issues?state=open"))return [];
            if(path.endsWith("/issues")&&args.method==="POST")
                return {number:101,html_url:"https://github.com/Kamara229/JessicaCore/issues/101"};
            return null;
        };
        const response=await startCodexDay("auto",api);
        assert.equal(response.ok,true);
        assert.equal(response.usesPaidModel,false);
        assert.equal(calls.length,3);
        assert.match(calls[1].args.body.title,/day:auto$/);
        assert.match(calls[2].path,/jessica-day.yml\/dispatches$/);
        assert.equal(calls[2].args.body.inputs.goal,"auto");
        assert.equal(calls[2].args.body.ref,"main");
    }finally{
        if(oldOwner===undefined)delete process.env.JESSICA_CODEX_OWNER_KEY;
        else process.env.JESSICA_CODEX_OWNER_KEY=oldOwner;
        if(oldGithub===undefined)delete process.env.JESSICA_GITHUB_AUTOMATION_TOKEN;
        else process.env.JESSICA_GITHUB_AUTOMATION_TOKEN=oldGithub;
    }
});
test("Free day workflow is constrained to safe, deterministic tests",()=>{
    const prepare=readFileSync(new URL("../../.github/day/prepare.mjs",import.meta.url),"utf8");
    const guard=readFileSync(new URL("../../.github/day/guard.mjs",import.meta.url),"utf8");
    const verify=readFileSync(new URL("../../.github/day/verify.mjs",import.meta.url),"utf8");
    assert.match(prepare,/regressionDayLanguage/);
    assert.match(prepare,/regressionDayContext/);
    assert.match(prepare,/regressionDayFoundations/);
    assert.match(guard,/modified.*HEAD/);
    assert.match(verify,/day:/);
});
