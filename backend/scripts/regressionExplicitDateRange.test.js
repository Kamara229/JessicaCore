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

test("same leap-day endpoints are two occurrences but zero elapsed days",async()=>{
  const {extractSameDayRange}=await import("../core/native/sameDayRange.js");
  const {decomposeTask}=await import("../core/taskDecomposer.js");
  const task="Сколько дней между 29.02.2024 и 29.02.2024? Исключи начальную дату, включи конечную. Покажи расчёт";
  assert.deepEqual(extractSameDayRange(task),{startDate:"2024-02-29",endDate:"2024-02-29"});
  const decomp=await decomposeTask({task,memoryContext:{skills:[]}});
  assert.equal(decomp.native,true);
  assert.equal(decomp.source,"native-day-range");
  assert.equal(decomp.decomposition.subtasks[0].text,task);
  const skill={id:"date-difference-calculation",enabled:true,status:"published",
    executionProgram:{mode:"native",version:1,strategy:"date_difference"},
    metadata:{bootstrap:{executable:true}},
    confidence:.9,learning:{maturityLevel:"READY",confidence:.9,occurrences:4}};
  const found=searchExperience(task,[skill]);
  assert.equal(found.found,true);
  assert.equal(found.experience.id,skill.id);
  const plan=tryCreateNativeSkillPlan({task,executionExperience:{skills:[skill]}});
  assert.equal(plan.handled,true);
  assert.deepEqual(plan.plan.steps[0].arguments,
    {startDate:"2024-02-29",endDate:"2024-02-29",mode:"elapsed"});
  assert.equal(plan.plan.steps[0].tool,"date_diff");
  const actual=await executeTool("date_diff",plan.plan.steps[0].arguments);
  assert.equal(actual.success,true);
  assert.equal(actual.data.fullDays,0);
  assert.match(actual.text,/0 полных календарных дней/);
});

test("same-day extraction rejects incomplete, invalid and extra-date ranges",async()=>{
  const {extractSameDayRange}=await import("../core/native/sameDayRange.js");
  assert.deepEqual(extractSameDayRange("Сколько дней между 2024-02-29 и 2024-02-29?"),
    {startDate:"2024-02-29",endDate:"2024-02-29"});
  for(const task of [
    "Сколько дней между 29.02.2024 и 01.03.2024?",
    "Сколько дней между 29.02.2025 и 29.02.2025?",
    "Сколько дней между 29.02.2024 и 29.02.2024 и 01.03.2024?",
    "Дата 29.02.2024; повтор 29.02.2024",
    "Сколько дней между 29.02.2024 и"
  ]) assert.equal(extractSameDayRange(task),null,task);
});

test("same named Russian leap-date endpoints avoid unavailable AI and return zero",async()=>{
 const task="Сколько дней между 29 февраля 2024 и 29 февраля 2024? Исключи начальную дату, включи конечную. Покажи расчёт";
 const {extractSameDayRange}=await import("../core/native/sameDayRange.js");
 const {decomposeTask}=await import("../core/taskDecomposer.js");
 const range={startDate:"2024-02-29",endDate:"2024-02-29"};
 assert.deepEqual(extractSameDayRange(task),range);
 assert.deepEqual(extractSameDayRange("Сколько дней между 29 февраля 2024 года и 29 февраля 2024 года?"),range);
 assert.deepEqual(extractSameDayRange("Сколько дней между 29.02.2024 и 29 февраля 2024?"),range);
 const decomposed=await decomposeTask({task,memoryContext:{skills:[]}});
 assert.equal(decomposed.native,true);
 assert.equal(decomposed.source,"native-day-range");
 assert.equal(decomposed.decomposition.subtasks[0].text,task);
 const skill={id:"date-difference-calculation",enabled:true,status:"published",
   executionProgram:{mode:"native",version:1,strategy:"date_difference"},
   metadata:{bootstrap:{executable:true}},learning:{maturityLevel:"READY"}};
 const found=searchExperience(task,[skill]);
 assert.equal(found.found,true);
 assert.equal(found.experience.id,skill.id);
 const plan=tryCreateNativeSkillPlan({task,executionExperience:{skills:[skill]}});
 assert.equal(plan.handled,true);
 assert.equal(plan.source,"experience:date-difference-calculation");
 assert.deepEqual(plan.plan.steps[0].arguments,{...range,mode:"elapsed"});
 const answer=await executeTool("date_diff",plan.plan.steps[0].arguments);
 assert.equal(answer.success,true);
 assert.equal(answer.data.fullDays,0);
 assert.match(answer.text,/Расч[её]т/u);
 assert.match(answer.text,/пустой интервал/u);
});

test("same-date named month guard rejects malformed or ambiguous inputs",async()=>{
 const {extractSameDayRange}=await import("../core/native/sameDayRange.js");
 for(const task of [
  "Сколько дней между 29 февраля 2025 и 29 февраля 2025?",
  "Сколько дней между 29 февраля 2024 и 1 марта 2024?",
  "Сколько дней между 29 фервраля 2024 и 29 фервраля 2024?",
  "Сколько дней между 29 февраля 2024 и 29 февраля 2024 и 1 марта 2024?",
  "29 февраля 2024; повтор 29 февраля 2024"
 ]) assert.equal(extractSameDayRange(task),null,task);
});
