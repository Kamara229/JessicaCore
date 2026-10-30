import "../tools/webFetchTool.js";
import "../tools/webSearchTool.js";
import test from "node:test";
import assert from "node:assert/strict";
import {tryCreateNativeSkillPlan} from "../core/planner/nativeSkillPlanner.js";

const request="Сравни Apache License 2.0 и MIT по патентным правам и их прекращению. Составь таблицу с условиями, короткими цитатами и официальными ссылками";
const readySourceSkill={
    id:"official-source-verification",name:"Проверка официального источника",
    enabled:true,status:"published",confidence:.9,
    learning:{maturityLevel:"READY",confidence:.9},
    executionProgram:{version:1,mode:"native",strategy:"official_source_lookup",evidenceMode:"source_content"}
};
const experience={found:true,used:true,skills:[readySourceSkill]};

test("two known licenses defeat READY generic one-source Skill in exact timeout case",()=>{
    const result=tryCreateNativeSkillPlan({
        task:request,executionExperience:experience,planningContext:{}
    });
    assert.equal(result.handled,true,result.validation?.text);
    assert.equal(result.plan.intent,"native_known_license_comparison");
    assert.equal(result.source,"canonical-license-texts");
    assert.deepEqual(result.plan.steps.map(step=>step.tool),["web_fetch","web_fetch"]);
    assert.deepEqual(result.plan.steps.map(step=>step.arguments.url),[
        "https://www.apache.org/licenses/LICENSE-2.0.txt",
        "https://opensource.org/license/mit"
    ]);
    assert.notEqual(result.plan.experience?.used,true);
});

test("generic source Skill still handles ordinary single-source lookup",()=>{
    const result=tryCreateNativeSkillPlan({
        task:"Найди официальный сайт Blender и подтверди ссылку",
        executionExperience:experience,planningContext:{}
    });
    assert.equal(result.handled,true,result.validation?.text);
    assert.equal(result.plan.intent,"native_official_source_verification");
    assert.equal(result.plan.steps.length,2);
});

test("unknown license names cannot be forced into two canonical URLs",()=>{
    const result=tryCreateNativeSkillPlan({
        task:"Сравни неизвестные лицензии Foo и Bar по патентным правам",
        executionExperience:{skills:[]},planningContext:{}
    });
    assert.equal(result.handled,false);
});
