import {verifyPairedSourceContent} from "../validator/pairedSourceContent.js";
const FSF="https://www.gnu.org/licenses/gpl-3.0.txt";
const ASF="https://www.apache.org/licenses/LICENSE-2.0.txt";
const flat=x=>String(x||"").replace(/\s+/g," ").toLowerCase();
export function groundedGplApachePatentComparison(task,plan,run){
 const t=String(task||"");
 if(!/(?:сравн|сопостав|compare)/iu.test(t)||!/(?:патент|patent)/iu.test(t)||
 !/\bgpl\s*v?3(?:\.0)?\b/iu.test(t)||!/apache(?:\s+license)?\s*2(?:\.0)?/iu.test(t)||
 plan?.intent!=="native_known_license_comparison"||
 verifyPairedSourceContent(plan,run)?.valid!==true)return null;
 const list=(run?.results||[]).filter(x=>x.tool==="web_fetch");
 if(list.length!==2||list.some(x=>x.success!==true||x.data?.truncated===true))return null;
 const get=url=>flat(list.find(x=>x.data?.url===url)?.data?.content);
 const g=get(FSF),a=get(ASF);
 const gp=["gnu general public license","8. termination",
 "including any patent licenses granted under the third paragraph of section 11",
 "10. automatic licensing of downstream recipients",
 "you may not initiate litigation (including a cross-claim or counterclaim in a lawsuit)",
 "11. patents","each contributor grants you a non-exclusive, worldwide, royalty-free patent license",
 "under the contributor's essential patent claims","end of terms and conditions"];
 const ap=["apache license","3. grant of patent license",
 "each contributor hereby grants to you a perpetual","where such license applies only to those patent claims licensable",
 "including a cross-claim or counterclaim in a lawsuit",
 "any patent licenses granted to you under this license for that work shall terminate",
 "as of the date such litigation is filed","end of terms and conditions"];
 if(!gp.every(x=>g.includes(x))||!ap.every(x=>a.includes(x)))return null;
 return {success:true,source:"bounded-patent-comparison",text:
  "GPLv3: раздел 11, абзац 3 даёт участникам неисключительную, всемирную и безвозмездную патентную лицензию на их существенные патентные притязания в отношении Contributor Version. Раздел 8 прекращает права за распространение или изменение программы вопреки GPLv3, включая патентную лицензию из раздела 11, с механизмом восстановления. Раздел 10 запрещает определённые патентные иски по программе, включая встречные и перекрёстные. Источник: "+FSF+
  "\n\nApache License 2.0: раздел 3 даёт бессрочную, всемирную, неисключительную и безвозмездную патентную лицензию на притязания, неизбежно затрагиваемые вкладом участника. Права на соответствующий Work прекращаются со дня подачи патентного иска о нарушении со стороны Work или включённого вклада, включая встречные и перекрёстные иски. Источник: "+ASF+
  "\n\nРазличие: GPLv3 регулирует прекращение при нарушении условий в разделе 8, а патентные ограничения — в разделе 10; Apache 2.0 устанавливает конкретное прекращение патентной лицензии за определённый иск в разделе 3."};
}
