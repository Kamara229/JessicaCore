const URL="https://www.gnu.org/licenses/gpl-3.0.txt";
const QUOTE=/conspicuously and appropriately publish on each copy an appropriate copyright notice/iu;
export function extractGplSection4Excerpt(task,plan,run){
 const q=String(task||"");
 if(plan?.intent!=="native_single_license_primary_text" || plan?.metadata?.licenseId!=="GPL-3.0" ||
   !/(?:§\s*4|section\s*4|раздел\s*4)/iu.test(q) ||
   !/(?:дословн|цитат|verbatim|quote)/iu.test(q) ||
   !/(?:уведомлен|авторск|copyright|notice)/iu.test(q)) return null;
 const steps=plan.steps||[],rows=run?.results||[];
 if(steps.length!==1||steps[0]?.arguments?.url!==URL||rows.length!==1||
   rows[0]?.tool!=="web_fetch"||rows[0]?.success!==true||rows[0]?.data?.url!==URL||
   rows[0]?.data?.truncated===true)return null;
 const source=String(rows[0].data.content||"").replace(/\s+/gu," ");
 if(!/gnu general public license/iu.test(source)||!/version 3, 29 june 2007/iu.test(source))return null;
 const start=/4\.\s*Conveying Verbatim Copies/iu.exec(source);
 if(!start)return null;
 const rest=source.slice(start.index);
 const end=/5\.\s*Conveying Modified Source Versions/iu.exec(rest);
 if(!end||end.index>4000)return null;
 const section=rest.slice(0,end.index);
 if(!/You may convey verbatim copies of the Program/iu.test(section)||
    !/keep intact all notices stating that this License/iu.test(section)||
    !/keep intact all notices of the absence of any warranty/iu.test(section)||
    !/give all recipients a copy of this License along with the Program/iu.test(section))return null;
 const match=QUOTE.exec(section);
 if(!match||match[0].split(/\s+/u).length!==12)return null;
 return {success:true,source:"bounded-gpl-section4-excerpt",text:"§4 GPLv3: “"+match[0]+"”.\nОфициальный URL: "+URL};
}
