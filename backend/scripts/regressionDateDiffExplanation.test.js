import test from "node:test";
import assert from "node:assert/strict";
import "../tools/dateDiffTool.js";
import {executeTool} from "../tools/toolRegistry.js";

test("the date calculation explains why October 9–23 means 14 elapsed days", async()=>{
    const result=await executeTool("date_diff",{
        startDate:"9 октября 2026",endDate:"23 октября 2026",mode:"elapsed"
    });
    assert.equal(result.success,true);
    assert.equal(result.data.fullDays,14);
    assert.match(result.text,/14 полных календарных дней/);
    assert.match(result.text,/2026-10-09 не учитывается/);
    assert.match(result.text,/2026-10-23 учитывается/);
});
test("calendar endpoints preserve no-off-by-one behavior", async()=>{
    const cases=[
        ["2026-10-22","2026-10-23",1],
        ["2026-10-23","2026-10-23",0],
        ["2024-02-28","2024-03-01",2],
        ["2024-02-29","2024-03-03",3]
    ];
    for(const [from,to,expected] of cases){
        const r=await executeTool("date_diff",{startDate:from,endDate:to,mode:"elapsed"});
        assert.equal(r.data?.fullDays,expected);
        assert.match(r.text,/Начальная дата/);
    }
});
test("absolute and backward elapsed modes preserve original contracts",async()=>{
    const abs=await executeTool("date_diff",{
        startDate:"2026-10-23",endDate:"2026-10-09",mode:"difference"
    });
    assert.equal(abs.success,true);
    assert.equal(abs.data.fullDays,14);
    const elapsed=await executeTool("date_diff",{
        startDate:"2026-10-23",endDate:"2026-10-09",mode:"elapsed"
    });
    assert.equal(elapsed.success,true);
    assert.equal(elapsed.data.chronologyValid,false);
    assert.equal(elapsed.data.signedDays,-14);
    assert.equal(elapsed.data.fullDays,null);
    assert.match(elapsed.text,/Расч[её]т:/u);
    assert.match(elapsed.text,/-14 календарных дней/u);
});

test("Russian calendar day endings follow 1/2–4/5+ grammar, including leap day",async()=>{
    const cases=[
        ["2024-02-28","2024-03-02",3,"3 полных календарных дня"],
        ["2024-02-29","2024-03-03",3,"3 полных календарных дня"],
        ["2026-10-01","2026-10-02",1,"1 полный календарный день"],
        ["2026-10-01","2026-10-03",2,"2 полных календарных дня"],
        ["2026-10-01","2026-10-06",5,"5 полных календарных дней"],
        ["2026-10-01","2026-10-12",11,"11 полных календарных дней"],
        ["2026-10-01","2026-10-22",21,"21 полный календарный день"],
        ["2026-10-01","2026-10-23",22,"22 полных календарных дня"],
        ["2026-10-01","2026-10-26",25,"25 полных календарных дней"]
    ];
    for(const [startDate,endDate,days,phrase] of cases){
        const r=await executeTool("date_diff",{startDate,endDate,mode:"elapsed"});
        assert.equal(r.success,true);
        assert.equal(r.data.fullDays,days);
        assert.ok(r.text.includes(phrase),r.text);
    }
});


test("cross-year elapsed explanation lists exactly 31 Dec and 1–3 Jan",async()=>{
  const result=await executeTool("date_diff",{
    startDate:"2026-12-30",endDate:"2027-01-03",mode:"elapsed"
  });
  assert.equal(result.success,true);
  assert.equal(result.data.fullDays,4);
  assert.match(result.text,/4 полных календарных дня/);
  assert.match(result.text,/2026-12-31, 2027-01-01, 2027-01-02, 2027-01-03/);
  assert.match(result.text,/Начальная дата 2026-12-30 не учитывается/);
});

test("Validator denies incomplete date explanations and Learning must not certify them",async()=>{
  const {validateResult}=await import("../core/validator.js");
  const task="Calculate the number of days excluding 30 December 2026 and explain the calculation";
  const plan={steps:[{id:"native_date_diff",tool:"date_diff"}],evidence:{mode:"none"}};
  const run={success:true,results:[{tool:"date_diff",success:true,data:{fullDays:4}}]};
  const incomplete=await validateResult(task,plan,run,{
    success:true,source:"tool",text:"Между 2026-12-30 и 2027-01-03 4 полных дня."
  });
  assert.equal(incomplete.valid,false);
  assert.equal(incomplete.failureScope,"presentation");
  const tool=await executeTool("date_diff",{startDate:"2026-12-30",endDate:"2027-01-03",mode:"elapsed"});
  const complete=await validateResult(task,plan,run,{
    success:true,source:"tool",text:tool.text
  });
  assert.equal(complete.valid,true);
});


test("Learning ignores a correct number without the requested date explanation", async()=>{
 const {analyzeExecutionTraceBatch}=await import("../core/learning/multiSkill/multiSkillAnalyzer.js");
 const {missingNativeDateExplanation}=await import("../core/learning/nativeDateExplanationEvidence.js");
 const task="Сколько дней с 30 декабря 2026 по 3 января 2027? Объясни расчёт";
 const trace={
   task,executedTools:["date_diff"],
   result:{success:true,verified:true,status:"COMPLETED",
     text:"Между 2026-12-30 и 2027-01-03 4 полных дня.",
     validation:{valid:true,degraded:false,outcomeType:"result"}},
   statistics:{completed:1},
   experienceUsage:{used:true,applied:true,appliedSkillIds:["date-difference-calculation"],
     skills:[{id:"date-difference-calculation"}]}
 };
 assert.equal(missingNativeDateExplanation(trace),true);
 const ignored=analyzeExecutionTraceBatch(trace);
 assert.equal(ignored[0].action,"IGNORE");
 assert.equal(ignored[0].metadata.positiveReinforcementAllowed,false);
 const complete=await executeTool("date_diff",{startDate:"2026-12-30",endDate:"2027-01-03",mode:"elapsed"});
 assert.equal(missingNativeDateExplanation({...trace,result:{...trace.result,text:complete.text}}),false);
});


test("next valid improvement excludes contaminated v5 evidence and keeps maturity below READY",async()=>{
  const {buildSkillImprovementContext}=await import("../core/learning/candidateBuilder/skillImprovement/skillImprovementContext.js");
  const task="Сколько дней с 30 декабря 2026 по 3 января 2027? Объясни расчёт";
  const good=await executeTool("date_diff",{
    startDate:"2026-12-30",endDate:"2027-01-03",mode:"elapsed"
  });
  const result=(text)=>({
    success:true,verified:true,status:"COMPLETED",text,
    validation:{valid:true,degraded:false,outcomeType:"result"}
  });
  const stored=Array.from({length:3},(_,i)=>({
    task,success:true,traceId:"verified-"+i,executedTools:["date_diff"],
    result:result(good.text)
  }));
  stored.push({
    task,success:true,traceId:"incomplete-legacy",executedTools:["date_diff"],
    result:result("Между 2026-12-30 и 2027-01-03 4 полных дня.")
  });
  const skill={
    id:"date-difference-calculation",name:"Date Difference",version:5,
    executionProgram:{mode:"native",version:1,strategy:"date_difference"},
    examples:stored,learning:{occurrences:4,confidence:.94}
  };
  const trace={
    id:"new-verified",task,executedTools:["date_diff"],
    result:result(good.text),statistics:{completed:1}
  };
  const candidate=buildSkillImprovementContext({
    skills:[skill],trace,occurrences:5,maturity:1,confidence:1
  });
  assert.equal(candidate.examples.length,4);
  assert.equal(candidate.metrics.successCount,4);
  assert.equal(candidate.metrics.occurrences,4);
  assert.equal(candidate.metrics.maturityLevel,"LEARNING");
  assert.equal(candidate.metrics.maturity,.88);
  assert.equal(candidate.metrics.confidence,.94);
  assert.ok(!candidate.examples.some(x=>x.traceId==="incomplete-legacy"));
});

test("single explicit two-date day range skips AI Decomposer and keeps original task",async()=>{
 const {decomposeTask}=await import("../core/taskDecomposer.js");
 const task="Сколько дней между 30.12.2026 и 03.01.2027? Исключи начальную дату, включи конечную. Покажи расчёт";
 const result=await decomposeTask({task,memoryContext:{skills:[]}});
 assert.equal(result.success,true);
 assert.equal(result.native,true);
 assert.equal(result.source,"native-day-range");
 assert.equal(result.decomposition.subtasks.length,1);
 assert.equal(result.decomposition.subtasks[0].text,task);
});
test("compound day tasks keep the existing decomposer, not the single-range shortcut",async()=>{
 const {tryNativeDecomposition}=await import("../core/decomposer/nativeDecomposer.js");
 for(const task of [
 "Сколько дней между 30.12.2026 и 03.01.2027? Затем найди календарь.",
 "Сколько дней между 30.12.2026 и 03.01.2027 и сколько часов?",
 "Сколько дней между 30.12.2026 и 03.01.2027 плюс ещё 5?"
 ]) assert.notEqual(tryNativeDecomposition({task,memoryContext:{skills:[]}}).source,"native-day-range");
});

test("same leap-day boundaries explain zero counted dates and retain verified Skill route",async()=>{
 const task="Сколько дней между 29.02.2024 и 29.02.2024? Исключи начальную дату, включи конечную. Покажи расчёт";
 const {tryCreateNativeSkillPlan}=await import("../core/planner/nativeSkillPlanner.js");
 const {validateResult}=await import("../core/validator.js");
 const planned=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]}});
 assert.equal(planned.handled,true);
 const tool=planned.plan.steps[0];
 assert.equal(tool.tool,"date_diff");
 assert.deepEqual(tool.arguments,
    {startDate:"2024-02-29",endDate:"2024-02-29",mode:"elapsed"});
 const actual=await executeTool("date_diff",tool.arguments);
 assert.equal(actual.success,true);
 assert.equal(actual.data.fullDays,0);
 assert.match(actual.text,/Расч[её]т:/u);
 assert.match(actual.text,/пустой интервал/u);
 assert.match(actual.text,/всего 0/u);
 assert.doesNotMatch(actual.text,/конечная дата 2024-02-29 учитывается/u);
 const verdict=await validateResult(task,planned.plan,
   {success:true,results:[{...actual,tool:"date_diff",id:tool.id}]},
   {success:true,source:"tool",text:actual.text});
 assert.equal(verdict.valid,true);
 assert.equal(verdict.degraded,false);
});

test("zero-day special wording does not alter other interval modes",async()=>{
 const normal=await executeTool("date_diff",{
   startDate:"2024-02-28",endDate:"2024-03-01",mode:"elapsed"
 });
 assert.match(normal.text,/2024-02-29, 2024-03-01/u);
 const difference=await executeTool("date_diff",{
   startDate:"2024-02-29",endDate:"2024-02-29",mode:"difference"
 });
 assert.equal(difference.data.fullDays,0);
 assert.doesNotMatch(difference.text,/пустой интервал/u);
});

test("reversed leap-day interval is a handled deterministic signed result without AI replanning",async()=>{
 const task="Сколько дней прошло между 03.03.2024 и 29.02.2024? Исключи начальную дату, включи конечную. Покажи расчёт";
 const {tryCreateNativeSkillPlan}=await import("../core/planner/nativeSkillPlanner.js");
 const {runPlan}=await import("../core/taskRunner.js");
 const {validateResult}=await import("../core/validator.js");
 const planned=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]}});
 assert.equal(planned.handled,true);
 assert.equal(planned.plan.steps[0].tool,"date_diff");
 assert.deepEqual(planned.plan.steps[0].arguments,{
   startDate:"2024-03-03",endDate:"2024-02-29",mode:"elapsed"
 });
 const executed=await runPlan(planned.plan,task);
 assert.equal(executed.success,true);
 const toolResult=executed.results[0];
 assert.equal(toolResult.success,true);
 assert.equal(toolResult.data.chronologyValid,false);
 assert.equal(toolResult.data.signedDays,-3);
 assert.equal(toolResult.data.fullDays,null);
 assert.match(toolResult.text,/Расч[её]т:/u);
 assert.match(toolResult.text,/-3 календарных дней/u);
 const validation=await validateResult(task,planned.plan,executed,{
   success:true,source:"tool",text:toolResult.text
 });
 assert.equal(validation.valid,true);
 assert.equal(validation.degraded,false);
});

test("Сколько дней прошло между routes backward explicit dates without AI Decomposer",async()=>{
  const {decomposeTask}=await import("../core/taskDecomposer.js");
  const {tryCreateNativeSkillPlan}=await import("../core/planner/nativeSkillPlanner.js");
  const {runPlan}=await import("../core/taskRunner.js");
  const original="Сколько дней прошло между 03.03.2024 и 29.02.2024? Исключи начальную дату, включи конечную. Покажи расчёт";
  const decomposition=await decomposeTask({task:original,memoryContext:{skills:[]}});
  assert.equal(decomposition.success,true);
  assert.equal(decomposition.native,true);
  assert.equal(decomposition.source,"native-day-range");
  assert.deepEqual(decomposition.decomposition.subtasks.map(s=>s.text),[original]);
  const plan=tryCreateNativeSkillPlan({task:original,executionExperience:{skills:[]}});
  assert.equal(plan.handled,true);
  assert.deepEqual(plan.plan.steps[0].arguments,{
    startDate:"2024-03-03",endDate:"2024-02-29",mode:"elapsed"
  });
  const executed=await runPlan(plan.plan,original);
  assert.equal(executed.success,true);
  assert.equal(executed.results[0].data.signedDays,-3);
  assert.match(executed.results[0].text,/-3 календарных дней/u);
});

test("elapsed-between native shortcut still rejects compound requests",async()=>{
  const {tryNativeDecomposition}=await import("../core/decomposer/nativeDecomposer.js");
  const make=task=>tryNativeDecomposition({task,memoryContext:{skills:[]}});
  for(const task of [
    "Сколько дней прошло между 03.03.2024 и 29.02.2024? Затем найди календарь.",
    "Сколько дней прошло между 03.03.2024 и 29.02.2024 и сколько часов?",
    "Сколько дней прошло между 03.03.2024 и 29.02.2024 плюс ещё 5?"
  ]) assert.notEqual(make(task)?.source,"native-day-range",task);
});

test("absolute backward date difference includes explicitly requested calculation",async()=>{
 const task="Сколько дней между 03.03.2024 и 29.02.2024? Рассчитай абсолютную разницу дат, независимо от их порядка. Покажи расчёт";
 const {tryCreateNativeSkillPlan}=await import("../core/planner/nativeSkillPlanner.js");
 const {runPlan}=await import("../core/taskRunner.js");
 const {validateResult}=await import("../core/validator.js");
 const planned=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]}});
 assert.equal(planned.handled,true);
 assert.deepEqual(planned.plan.steps[0].arguments,{
   startDate:"2024-03-03",endDate:"2024-02-29",mode:"difference"
 });
 const executed=await runPlan(planned.plan,task);
 assert.equal(executed.success,true);
 const actual=executed.results[0];
 assert.equal(actual.success,true);
 assert.equal(actual.data.fullDays,3);
 assert.equal(actual.data.signedDays,-3);
 assert.match(actual.text,/Расч[её]т:/u);
 assert.match(actual.text,/\|2024-02-29 − 2024-03-03\| = 3 полных дня/u);
 const ok=await validateResult(task,planned.plan,executed,
   {success:true,source:"tool",text:actual.text});
 assert.equal(ok.valid,true);
 assert.equal(ok.degraded,false);
 const incomplete="Между 2024-03-03 и 2024-02-29 3 полных дня.";
 const no=await validateResult(task,planned.plan,executed,
   {success:true,source:"tool",text:incomplete});
 assert.equal(no.valid,false);
 assert.equal(no.failureScope,"presentation");
 const {missingNativeDateExplanation}=await import("../core/learning/nativeDateExplanationEvidence.js");
 const trace={task,executedTools:["date_diff"],result:{text:incomplete}};
 assert.equal(missingNativeDateExplanation(trace),true);
 assert.equal(missingNativeDateExplanation({...trace,result:{text:actual.text}}),false);
 const forward=await executeTool("date_diff",{
   startDate:"2024-02-29",endDate:"2024-03-03",mode:"difference"
 });
 assert.equal(forward.data.fullDays,3);
 assert.match(forward.text,/Расч[её]т:/u);
 const equal=await executeTool("date_diff",{
   startDate:"2024-02-29",endDate:"2024-02-29",mode:"difference"
 });
 assert.equal(equal.data.fullDays,0);
 assert.match(equal.text,/\|2024-02-29 − 2024-02-29\| = 0 полных дней/u);
});

test("long elapsed requests still get a visible calculation",async()=>{
 const task="Сколько дней между 09.10.2026 и 23.10.2026? Исключи начальную дату, включи конечную. Покажи расчёт";
 const {tryCreateNativeSkillPlan}=await import("../core/planner/nativeSkillPlanner.js");
 const {validateResult}=await import("../core/validator.js");
 const plan=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]}}).plan;
 const date=await executeTool("date_diff",plan.steps[0].arguments);
 assert.equal(date.data.fullDays,14);
 assert.match(date.text,/Расч[её]т: \(2026-10-23 − 2026-10-09\) = 14 полных календарных дней/u);
 const run={success:true,results:[{...date,tool:"date_diff"}]};
 const verdict=await validateResult(task,plan,run,{success:true,source:"tool",text:date.text});
 assert.equal(verdict.valid,true);
});

test("invalid leap date returns a grounded clarification without AI planner or fake arithmetic",async()=>{
 const task="Сколько дней между 29.02.2025 и 01.03.2025? Покажи расчёт";
 const {extractInvalidCalendarRange}=await import("../core/native/invalidCalendarRange.js");
 const {decomposeTask}=await import("../core/taskDecomposer.js");
 const {tryCreateNativeSkillPlan}=await import("../core/planner/nativeSkillPlanner.js");
 const {runPlan}=await import("../core/taskRunner.js");
 const {validateResult}=await import("../core/validator.js");
 const range=extractInvalidCalendarRange(task);
 assert.deepEqual(range,{
   startDate:"29.02.2025",endDate:"01.03.2025",invalidDates:["29.02.2025"]
 });
 const d=await decomposeTask({task,memoryContext:{skills:[]}});
 assert.equal(d.success,true);
 assert.equal(d.native,true);
 assert.equal(d.source,"native-day-range");
 const planner=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]}});
 assert.equal(planner.handled,true);
 assert.equal(planner.source,"native-invalid-calendar-date");
 assert.equal(planner.plan.experience.used,false);
 assert.equal(planner.plan.steps.length,1);
 assert.equal(planner.plan.steps[0].tool,"invalid_calendar_date");
 const executed=await runPlan(planner.plan,task);
 assert.equal(executed.success,true);
 assert.equal(executed.results.length,1);
 assert.equal(executed.results[0].success,true);
 assert.equal(executed.results[0].data.calendarValid,false);
 assert.equal(executed.results[0].data.fullDays,null);
 assert.deepEqual(executed.results[0].data.invalidDates,["29.02.2025"]);
 assert.match(executed.results[0].text,/29\.02\.2025 не существует/u);
 assert.match(executed.results[0].text,/Расч[её]т количества дней невозможен/u);
 assert.doesNotMatch(executed.results[0].text,/прошло \d+ полных/u);
 const check=await validateResult(task,planner.plan,executed,{
   success:true,source:"tool",text:executed.results[0].text
 });
 assert.equal(check.valid,true);
 assert.equal(check.degraded,false);
});
test("calendar guard rejects false positives, extra dates, compound tasks and valid leap dates",async()=>{
 const {extractInvalidCalendarRange}=await import("../core/native/invalidCalendarRange.js");
 assert.equal(extractInvalidCalendarRange("Сколько дней между 29.02.2024 и 01.03.2024? Покажи расчёт"),null);
 const leap=extractInvalidCalendarRange("Сколько дней между 29 февраля 2025 и 1 марта 2025? Покажи расчёт");
 assert.deepEqual(leap?.invalidDates,["29 февраля 2025"]);
 assert.deepEqual(extractInvalidCalendarRange("Сколько дней между 2025-02-29 и 2025-03-01? Покажи расчёт")?.invalidDates,
   ["2025-02-29"]);
 for(const task of [
  "Сколько дней между 29.02.2025 и 01.03.2025? Затем проверь официальный календарь.",
  "Сколько дней между 29.02.2025 и 01.03.2025 и сколько часов?",
  "Сколько дней между 29.02.2025 и 01.03.2025? Покажи расчёт для 01.03.2026",
  "Сколько дней между 29.02.2025 и 01.03.2025? Подскажи погоду."
 ]) assert.equal(extractInvalidCalendarRange(task),null,task);
 const {tryCreateNativeSkillPlan}=await import("../core/planner/nativeSkillPlanner.js");
 assert.equal(tryCreateNativeSkillPlan({
  task:"Сколько дней между 28.02.2025 и 01.03.2025? Покажи расчёт",
  executionExperience:{skills:[]}
 }).plan.steps[0].tool,"date_diff");
});

test("nonexistent calendar date bypasses AI Composer and Validator with direct proof",async()=>{
 const task="Сколько дней между 29.02.2025 и 01.03.2025? Покажи расчёт";
 const {decomposeTask}=await import("../core/taskDecomposer.js");
 const {tryCreateNativeSkillPlan}=await import("../core/planner/nativeSkillPlanner.js");
 const {runPlan}=await import("../core/taskRunner.js");
 const {composeAnswer}=await import("../core/answerComposer.js");
 const {getDirectAnswer}=await import("../core/composer/directAnswer.js");
 const {validateResult}=await import("../core/validator.js");
 const d=await decomposeTask({task,memoryContext:{skills:[]}});
 assert.equal(d.source,"native-day-range");
 const native=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]}});
 assert.equal(native.source,"native-invalid-calendar-date");
 assert.equal(native.plan.experience.used,false);
 const result=await runPlan(native.plan,task);
 assert.equal(result.success,true);
 assert.equal(result.results[0].data.calendarValid,false);
 const direct=getDirectAnswer(result);
 assert.equal(direct.source,"tool");
 assert.equal(direct.tool,"invalid_calendar_date");
 const answer=await composeAnswer(task,native.plan,result);
 assert.deepEqual(answer,direct);
 assert.ok(answer.text.includes("29.02.2025 не существует"));
 const verdict=await validateResult(task,native.plan,result,answer);
 assert.equal(verdict.valid,true);
 assert.equal(verdict.degraded,false);
 assert.equal(verdict.validationMode,"tool_direct");
 assert.equal(getDirectAnswer({results:[{...result.results[0],success:false}]}),null);
});

test("invalid calendar input is not positive Experience or Pattern Discovery evidence",async()=>{
 const {skipInvalidCalendarInputLearning}=await import("../core/learning/nativeCalendarInputLearningGate.js");
 const {processExecutionLearning}=await import("../core/learning/learningFacade.js");
 const {analyzeExecutionTraceBatch}=await import("../core/learning/multiSkill/multiSkillAnalyzer.js");
 const task="Сколько дней между 31.04.2026 и 01.05.2026? Покажи расчёт";
 const trace={
   task,executedTools:["invalid_calendar_date"],
   status:"COMPLETED",
   result:{success:true,status:"COMPLETED",verified:true,
     validation:{valid:true,degraded:false,outcomeType:"result"},
     text:"Дата 31.04.2026 не существует в календаре. Расчёт количества дней невозможен. Уточните дату."},
   statistics:{completed:1},
   experienceUsage:{found:false,used:false,applied:false,skills:[]}
 };
 assert.equal(skipInvalidCalendarInputLearning(trace),true);
 const learning=await processExecutionLearning(trace);
 assert.equal(learning.success,true);
 assert.equal(learning.queued,false);
 assert.equal(learning.skipped,true);
 assert.equal(learning.reason,"native_invalid_calendar_input");
 assert.equal(skipInvalidCalendarInputLearning({...trace,executedTools:[]}),false);
 assert.equal(skipInvalidCalendarInputLearning({...trace,executedTools:["invalid_calendar_date","date_diff"]}),false);
 assert.equal(skipInvalidCalendarInputLearning({...trace,task:"Сколько дней между 28.04.2026 и 01.05.2026? Покажи расчёт"}),false);
 assert.equal(skipInvalidCalendarInputLearning({...trace,task:"Сколько дней между 31.04.2026 и 01.05.2026? Затем проверь календарь."}),false);
});

test("a multiply-then-add request resolves through verified native execution without AI",async()=>{
 const task="Вычисли 123 × 456 и прибавь 7. Покажи расчёт";
 const {decomposeTask}=await import("../core/taskDecomposer.js");
 const {tryCreateNativeSkillPlan}=await import("../core/planner/nativeSkillPlanner.js");
 const {runPlan}=await import("../core/taskRunner.js");
 const {composeAnswer}=await import("../core/answerComposer.js");
 const {validateResult}=await import("../core/validator.js");
 const {verifiedNativeIntegerProductAdd}=await import("../core/validator/nativeIntegerProductAddProof.js");
 const decomp=await decomposeTask({task,memoryContext:{skills:[]}});
 assert.equal(decomp.native,true);
 assert.equal(decomp.source,"native-integer-product-add");
 assert.deepEqual(decomp.decomposition.subtasks.map(x=>x.text),[task]);
 const planned=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]}});
 assert.equal(planned.handled,true);
 assert.equal(planned.source,"native-integer-product-add");
 assert.equal(planned.plan.experience.used,false);
 assert.equal(planned.plan.steps.length,1);
 assert.equal(planned.plan.steps[0].tool,"integer_multiply_add");
 assert.deepEqual(planned.plan.steps[0].arguments,{a:"123",b:"456",c:"7",showWork:true});
 const executed=await runPlan(planned.plan,task);
 assert.equal(executed.success,true);
 assert.equal(executed.results[0].data.product,"56088");
 assert.equal(executed.results[0].data.sum,"56095");
 const answer=await composeAnswer(task,planned.plan,executed);
 assert.equal(answer.success,true);
 assert.equal(answer.source,"native-integer-multiply-add-proof");
 assert.match(answer.text,/123 × 456 = 56088/u);
 assert.match(answer.text,/56088 \+ \(7\) = 56095/u);
 assert.match(answer.text,/Ответ: 56095/u);
 const verdict=await validateResult(task,planned.plan,executed,answer);
 assert.equal(verdict.valid,true);
 assert.equal(verdict.degraded,false);
 assert.equal(verdict.validationMode,"tool_direct");
 assert.equal(verifiedNativeIntegerProductAdd(task,planned.plan,executed,{...answer,text:"Ответ: 56096"}),false);
 assert.equal(verifiedNativeIntegerProductAdd(task,planned.plan,{...executed,
  results:[{...executed.results[0],data:{product:"56088",sum:"56096"}}]},answer),false);
});

test("multiply-then-add uses exact integer signs and rejects ambiguous extra operations",async()=>{
 const {parseIntegerProductAddTask}=await import("../core/native/integerProductAddIntent.js");
 const {multiplyAndAddIntegers}=await import("../tools/integerProductAddTool.js");
 for(const [task,sum] of [
   ["Вычисли 2 × 3 и прибавь 0. Покажи расчёт","6"],
   ["Вычисли (-125) × (-32) и прибавь (-7). Покажи расчёт","3993"],
   ["Вычисли 999999999999999 × 999999999999999 и прибавь 1. Покажи расчёт",
     "999999999999998000000000000002"]
 ]){
   const p=parseIntegerProductAddTask(task);
   assert.ok(p,task);
   assert.equal(multiplyAndAddIntegers(p).data.sum,sum);
 }
 for(const task of [
   "Вычисли 123 × 456 и прибавь 7. Затем умножь на 2",
   "Вычисли 123 × 456 и прибавь 7 и вычти 2",
   "Вычисли 123 × 456 и прибавь 7. Покажи расчёт 8 × 9",
   "Вычисли 123 × 456 и прибавь 7. Подскажи погоду",
   "Вычисли 123 × 456 и прибавь (7. Покажи расчёт"
 ])assert.equal(parseIntegerProductAddTask(task),null,task);
 const {parseIntegerProductTask}=await import("../core/native/integerMultiplicationIntent.js");
 assert.equal(parseIntegerProductTask("Вычисли 123 × 456 и прибавь 7. Покажи расчёт"),null);
});
