import {verifyPairedSourceContent} from "../validator/pairedSourceContent.js";

const GPL="https://www.gnu.org/licenses/gpl-3.0.txt";
const APACHE="https://www.apache.org/licenses/LICENSE-2.0.txt";
const flat=x=>String(x||"").replace(/\s+/gu," ").toLowerCase();

/** Compare patent suit restrictions from the exact requested numbered sections. */
export function gpl10PatentRestrictions(task,plan,run) {
 const q=String(task||"");
 if(plan?.intent!=="native_known_license_comparison" ||
    !/(?:сравн|сопостав|compare)/iu.test(q) ||
    !/(?:патент|patent)/iu.test(q) ||
    !/\bgpl\s*v?3(?:\.0)?\b/iu.test(q) ||
    !/apache(?:\s+license)?\s*2(?:\.0)?/iu.test(q) ||
    !/(?:§\s*10|раздел\s*10|section\s*10)/iu.test(q) ||
    !/(?:§\s*3|раздел\s*3|section\s*3)/iu.test(q) ||
    verifyPairedSourceContent(plan,run)?.valid!==true)return null;
 const rows=(run?.results||[]).filter(x=>x?.tool==="web_fetch");
 if(rows.length!==2||rows.some(x=>x.success!==true||x.data?.truncated===true))return null;
 const body=url=>flat(rows.find(x=>x.data?.url===url)?.data?.content);
 const g=body(GPL),a=body(APACHE);
 const gi=g.indexOf("10. automatic licensing of downstream recipients.");
 const ge=g.indexOf("11. patents.",gi);
 const ai=a.indexOf("3. grant of patent license.");
 const ae=a.indexOf("4. redistribution.",ai);
 if(gi<0||ge<gi||ge-gi>4500||ai<0||ae<ai||ae-ai>4500)return null;
 const g10=g.slice(gi,ge),a3=a.slice(ai,ae);
 const requiredG=[
  "each time you convey a covered work",
  "you may not impose any further restrictions",
  "you may not initiate litigation (including a cross-claim or counterclaim in a lawsuit)",
  "alleging that any patent claim is infringed by making, using, selling, offering for sale, or importing the program or any portion of it"
 ];
 const requiredA=[
  "each contributor hereby grants to you a perpetual",
  "if you institute patent litigation against any entity",
  "including a cross-claim or counterclaim in a lawsuit",
  "alleging that the work or a contribution incorporated within the work",
  "constitutes direct or contributory patent infringement",
  "any patent licenses granted to you under this license for that work shall terminate",
  "as of the date such litigation is filed"
 ];
 if(!requiredG.every(x=>g10.includes(x))||!requiredA.every(x=>a3.includes(x)))return null;
 return {success:true,source:"bounded-patent-comparison",text:
  "GPLv3 §10: получатель не вправе вводить дополнительные ограничения предоставленных лицензией прав, в том числе инициировать патентный иск (включая встречный или перекрёстный), утверждающий, что изготовление, использование, продажа, предложение к продаже или импорт полученной Программы либо её части нарушает патент. Это ограничение касается именно Программы, а не любых патентных споров. За нарушение GPLv3 последствия прекращения прав предусмотрены отдельно в §8. Источник: "+GPL+
  "\n\nApache License 2.0 §3: если лицензиат подаёт патентный иск (включая встречный либо перекрёстный), утверждающий, что Work или включённый в него вклад прямо либо косвенно нарушает патент, то предоставленные по Apache патентные лицензии в отношении этого Work прекращаются с даты подачи иска. §3 описывает последствие определённого иска, а не общий запрет на любые патентные иски. Источник: "+APACHE+
  "\n\nРазличие: GPLv3 §10 прямо ограничивает инициирование указанных патентных исков против полученной Программы; Apache 2.0 §3 прямо прекращает патентную лицензию на соответствующий Work при подаче определённого иска."};
}
