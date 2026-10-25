import test from "node:test";
import assert from "node:assert/strict";
import "../tools/dateDiffTool.js";
import { executeTool } from "../tools/toolRegistry.js";
import { tryCreateNativeSkillPlan } from "../core/planner/nativeSkillPlanner.js";
import { extractExplicitDateRange } from "../core/native/explicitDateRange.js";
import { searchExperience } from "../experience/search/experienceSearch.js";

test("explicit excluded 27 February endpoint is not advanced twice", async () => {
    const task = "Сколько дней с 27 февраля по 2 марта 2024 года (27-е исключить, 2-е включить)?";
    const planned = tryCreateNativeSkillPlan({task, executionExperience:{skills:[]}});
    assert.equal(planned.handled,true);
    const step = planned.plan.steps[0];
    assert.equal(step.tool,"date_diff");
    assert.deepEqual(step.arguments,{
        startDate:"2024-02-27",endDate:"2024-03-02",mode:"elapsed"
    });
    const actual = await executeTool("date_diff", step.arguments);
    assert.equal(actual.success,true);
    assert.equal(actual.data.fullDays,4);
    assert.match(actual.text,/4 полных календарных дня/);
    assert.match(actual.text,/2024-02-27 не учитывается/);
});

test("explicit date range preserves leap and year rollover boundaries",()=>{
    assert.deepEqual(
        extractExplicitDateRange("с 28 февраля по 1 марта 2024 года"),
        {startDate:"2024-02-28",endDate:"2024-03-01"}
    );
    assert.deepEqual(
        extractExplicitDateRange("с 29 декабря по 2 января 2024 года"),
        {startDate:"2023-12-29",endDate:"2024-01-02"}
    );
    assert.equal(extractExplicitDateRange("Дата — 2 марта 2024 года"),null);
});


test("decomposed between-and instruction stays executable during AI 429", async () => {
    const original="Сколько дней прошло с 25 октября по 2 ноября 2026 года, если 25 октября исключить, а 2 ноября включить? Объясни расчёт.";
    const rewritten="Вычислить количество дней между 25 октября и 2 ноября 2026 года, исключая 25 октября и включая 2 ноября, и объяснить, как получено число.";
    for (const task of [original,rewritten]) {
        const result=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]}});
        assert.equal(result.handled,true,task);
        assert.equal(result.plan.steps.length,1);
        assert.deepEqual(result.plan.steps[0].arguments,{
            startDate:"2026-10-25",endDate:"2026-11-02",mode:"elapsed"
        });
        const executed=await executeTool("date_diff",result.plan.steps[0].arguments);
        assert.equal(executed.success,true);
        assert.equal(executed.data.fullDays,8);
        assert.match(executed.text,/8 полных календарных дней/);
    }
});

test("cross-month date intent retrieves native date Skill without claiming use prematurely",()=>{
    const skill={id:"date-difference-calculation",enabled:true,status:"published",
        executionProgram:{mode:"native",version:1,strategy:"date_difference"},
        learning:{maturityLevel:"LEARNING",confidence:.82,successCount:2},
        examples:[{success:true,executedTools:["date_diff"],result:{
            success:true,verified:true,status:"COMPLETED",validation:{valid:true,degraded:false}
        }}]};
    const task="Определить количество дней между 27 октября и 3 ноября 2026 года, исключая 27 октября";
    const found=searchExperience(task,[skill]);
    assert.equal(found.found,true);
    assert.equal(found.experience,skill);
    const planned=tryCreateNativeSkillPlan({task,executionExperience:{skills:[found.experience]}});
    assert.equal(planned.source,"experience:date-difference-calculation");
    assert.equal(planned.plan.steps[0].tool,"date_diff");
    assert.equal(searchExperience("Объясни основы бухучёта",[skill]).found,false);
});


test("English decomposition retains excluded December start and Russian count explanation",async()=>{
  const utterances=[
    "Сколько дней с 30 декабря 2026 по 3 января 2027, исключая 30 декабря? Объясни расчёт",
    "Calculate the number of days between 30 December 2026 and 3 January 2027, excluding 30 December, and explain the calculation."
  ];
  for(const task of utterances){
    const plan=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]}});
    assert.equal(plan.handled,true,task);
    assert.equal(plan.plan.steps[0].arguments.mode,"elapsed");
    assert.equal(plan.plan.steps[0].arguments.startDate,"2026-12-30");
    assert.equal(plan.plan.steps[0].arguments.endDate,"2027-01-03");
    const reply=await executeTool("date_diff",plan.plan.steps[0].arguments);
    assert.equal(reply.data?.fullDays,4);
    assert.match(reply.text,/2026-12-31, 2027-01-01, 2027-01-02, 2027-01-03/);
  }
});
