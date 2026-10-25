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
