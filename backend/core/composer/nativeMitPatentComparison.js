import {verifyPairedSourceContent} from "../validator/pairedSourceContent.js";
const APACHE="https://www.apache.org/licenses/LICENSE-2.0.txt",MIT="https://opensource.org/license/mit";
const flat=s=>String(s||"").replace(/\s+/g," ").toLowerCase();
export function groundedMitPatentComparison(task,plan,run){
 const text=String(task||"");
 if(!/сравн|сопостав|compare/iu.test(text)||!/патент|patent/iu.test(text)||
    !/apache(?:\s+license)?\s*2(?:\.0)?/iu.test(text)||!/\bmit\b/iu.test(text)||
    plan?.intent!=="native_known_license_comparison"||
    verifyPairedSourceContent(plan,run)?.valid!==true)return null;
 const fetched=(run?.results||[]).filter(x=>x?.tool==="web_fetch"&&x.success===true);
 const a=fetched.find(x=>x.data?.url===APACHE),m=fetched.find(x=>x.data?.url===MIT);
 if(fetched.length!==2||!a||!m||a.data.truncated||m.data.truncated)return null;
 const apache=flat(a.data.content),mit=flat(m.data.content);
 const starts=mit.indexOf("permission is hereby granted, free of charge");
 const ends=mit.indexOf("other dealings in the software",starts);
 if(starts<0||ends<starts+400)return null;
 const operative=mit.slice(starts,ends+30);
 const clauses=[
  "3. grant of patent license",
  "each contributor hereby grants to you a perpetual",
  "patent license to make, have made, use",
  "if you institute patent litigation against any entity",
  "including a cross-claim or counterclaim in a lawsuit",
  "alleging that the work or a contribution incorporated within the work",
  "constitutes direct or contributory patent infringement",
  "patent licenses granted to you under this license for that work shall terminate",
  "as of the date such litigation is filed"
 ];
 const mitParts=[
  "including without limitation the rights to use, copy, modify, merge, publish, distribute",
  "the above copyright notice and this permission notice shall be included",
  "the software is provided",
  "in no event shall the authors or copyright holders be liable"
 ];
 if(!clauses.every(q=>apache.includes(q))||!mitParts.every(q=>operative.includes(q))||
    /\bpatent\b/u.test(operative))return null;
 return {success:true,source:"bounded-patent-comparison",text:
  "Apache 2.0: раздел 3 прямо предоставляет патентную лицензию и предусматривает прекращение патентных прав для соответствующего Work со дня подачи иска, утверждающего прямое либо содействующее нарушение патента произведением или включённым вкладом (в том числе встречного или перекрёстного иска), но не при любом патентном споре. Источник: "+APACHE+
  "\n\nMIT License: в полном опубликованном тексте нет прямо сформулированной патентной лицензии или условия её прекращения. Текст MIT не имеет нумерованных разделов; права предоставляются в первом абзаце, далее следуют условие об уведомлениях и отказ от гарантий. Это не исключает возможных подразумеваемых патентных прав. Источник: "+MIT};
}
