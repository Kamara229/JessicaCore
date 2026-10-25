/**
 * Reproduce a narrow original-language quotation only after matching
 * it against the just-retrieved official document.
 */
const URL="https://www.apache.org/licenses/LICENSE-2.0.txt";
const FRAGMENT=/licenses granted to You under this License for that Work shall terminate/u;

export function extractOfficialLicenseExcerpt(task,plan,run){
  const query=String(task||"");
  if(plan?.intent!=="native_single_license_primary_text" ||
     plan?.metadata?.licenseId!=="Apache-2.0" ||
     !/(?:§\s*3|section\s*3|раздел\s*3)/iu.test(query) ||
     !/(?:патент|patent)/iu.test(query) ||
     !/(?:английск|english)/iu.test(query) ||
     !/(?:дословн|цитат|фраз|формулировк|скопир|копиров|приведи\s+\d{1,2}\s+английск|verbatim|quote)/iu.test(query))
      return null;
  const steps=Array.isArray(plan.steps)?plan.steps:[];
  const rows=Array.isArray(run?.results)?run.results:[];
  if(steps.length!==1 || steps[0]?.tool!=="web_fetch" ||
     steps[0]?.arguments?.url!==URL || rows.length!==1 ||
     rows[0]?.tool!=="web_fetch" || rows[0]?.success!==true ||
     rows[0]?.data?.url!==URL || rows[0]?.data?.truncated===true)
      return null;
  const body=String(rows[0]?.data?.content||"").replace(/\s+/gu," ");
  const start=/3\.\s*Grant of Patent License/iu.exec(body);
  if(!start)return null;
  const remaining=body.slice(start.index);
  const finish=/4\.\s*Redistribution/iu.exec(remaining);
  if(!finish||finish.index>6000)return null;
  const section=remaining.slice(0,finish.index);
  if(!section.includes("If You institute patent litigation") ||
     !section.includes("as of the date such litigation is filed"))
      return null;
  const match=FRAGMENT.exec(section);
  if(!match || match[0].split(/\s+/u).length!==12)return null;
  return {
    success:true,source:"bounded-original-license-excerpt",
    text:"§3 Apache License 2.0: “"+match[0]+"”.\nОфициальный URL: "+URL
  };
}
