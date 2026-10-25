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
    assert.equal(elapsed.success,false);
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
