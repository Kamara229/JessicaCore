/**
 * Answer a single-license "which section grants patent rights?" question
 * from newly fetched official Apache text. Do not synthesize legal claims.
 */
const URL="https://www.apache.org/licenses/LICENSE-2.0.txt";

export function groundedApachePatentSection(task,plan,run){
    const t=String(task||"");
    if(plan?.intent!=="native_single_license_primary_text" ||
       plan?.metadata?.licenseId!=="Apache-2.0" ||
       !/\bApache(?:\s+License)?\s*2(?:\.0)?\b/iu.test(t) ||
       !/(?:патент|patent)/iu.test(t) ||
       !/(?:пункт|раздел|секци|section|§|where|which|укажи|назови)/iu.test(t) ||
       !/(?:лицензи|license|grant)/iu.test(t) ||
       /(?:дословн|английских\s+слов|цитат|скопир|verbatim|quote)/iu.test(t))
       return null;
    const steps=Array.isArray(plan.steps)?plan.steps:[];
    const rows=Array.isArray(run?.results)?run.results:[];
    if(steps.length!==1 || steps[0]?.tool!=="web_fetch" ||
       steps[0]?.arguments?.url!==URL ||
       rows.length!==1 || rows[0]?.tool!=="web_fetch" ||
       rows[0]?.success!==true || rows[0]?.data?.url!==URL ||
       rows[0]?.data?.truncated===true)return null;

    const body=String(rows[0]?.data?.content||"").replace(/\s+/gu," ");
    const start=/\b3\.\s*Grant of Patent License\./iu.exec(body);
    if(!start)return null;
    const remaining=body.slice(start.index);
    const end=/\b4\.\s*Redistribution\./iu.exec(remaining);
    if(!end || end.index>5500)return null;
    const clause=remaining.slice(0,end.index);
    if(!/Each Contributor hereby grants to You a perpetual/iu.test(clause) ||
       !/patent license to make, have made, use/iu.test(clause) ||
       !/If You institute patent litigation against any entity/iu.test(clause) ||
       !/for that Work shall terminate as of the date such litigation is filed/iu.test(clause))
       return null;
    return {
        success:true,
        source:"bounded-apache-patent-section",
        text:"Патентная лицензия Apache License 2.0 предусмотрена в разделе 3 — «Grant of Patent License» («Предоставление патентной лицензии»).\nОфициальный текст: "+URL
    };
}
