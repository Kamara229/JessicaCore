import test from "node:test";
import assert from "node:assert/strict";
import "../tools/dateDiffTool.js";
import { executeTool } from "../tools/toolRegistry.js";
import { tryCreateNativeSkillPlan } from "../core/planner/nativeSkillPlanner.js";
import { extractExplicitDateRange } from "../core/native/explicitDateRange.js";

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
