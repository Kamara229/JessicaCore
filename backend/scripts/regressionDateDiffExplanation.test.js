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
        ["2024-02-28","2024-03-01",2]
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
    assert.equal(elapsed.success,false);
});
