import {verifyPairedSourceContent} from "../validator/pairedSourceContent.js";
const GPL="https://www.gnu.org/licenses/gpl-3.0.txt";
const APACHE="https://www.apache.org/licenses/LICENSE-2.0.txt";
const flatten=value=>String(value||"").replace(/\s+/gu," ").toLowerCase();

export function compareLicenseRestoration(task,plan,run){
 const query=String(task||"");
 if(plan?.intent!=="native_known_license_comparison" ||
    !/(?:сравн|сопостав|compare)/iu.test(query) ||
    !/(?:восстанов|возобнов|reinstat|restore)/iu.test(query) ||
    !/\bgpl\s*v?3(?:\.0)?\b/iu.test(query) ||
    !/apache(?:\s+license)?\s*2(?:\.0)?/iu.test(query) ||
    !/(?:§\s*8|раздел\s*8|section\s*8)/iu.test(query) ||
    !/(?:§\s*3|раздел\s*3|section\s*3)/iu.test(query) ||
    verifyPairedSourceContent(plan,run)?.valid!==true) return null;
 const results=(run?.results||[]).filter(x=>x.tool==="web_fetch");
 if(results.length!==2||results.some(x=>x.success!==true||x.data?.truncated===true))return null;
 const body=url=>flatten(results.find(x=>x.data?.url===url)?.data?.content);
 const g=body(GPL),a=body(APACHE);
 const gs=g.indexOf("8. termination."),ge=g.indexOf("9. acceptance not required",gs);
 const as=a.indexOf("3. grant of patent license."),ae=a.indexOf("4. redistribution.",as);
 if(gs<0||ge<gs||ge-gs>4500||as<0||ae<as||ae-as>4500)return null;
 const g8=g.slice(gs,ge),a3=a.slice(as,ae);
 const requiredG=[
  "including any patent licenses granted under the third paragraph of section 11",
  "reinstated (a) provisionally",
  "prior to 60 days after the cessation",
  "this is the first time you have received notice of violation",
  "you cure the violation prior to 30 days after your receipt of the notice"
 ];
 const requiredA=[
  "each contributor hereby grants to you a perpetual",
  "if you institute patent litigation against any entity",
  "constitutes direct or contributory patent infringement",
  "any patent licenses granted to you under this license for that work shall terminate",
  "as of the date such litigation is filed"
 ];
 if(!requiredG.every(x=>g8.includes(x))||!requiredA.every(x=>a3.includes(x)))return null;
 return {success:true,source:"bounded-patent-comparison",text:
  "GPLv3 §8: нарушение условий лицензии автоматически прекращает права, включая патентные лицензии, предоставленные по третьему абзацу §11. После прекращения нарушения лицензия от конкретного правообладателя восстанавливается предварительно, если он не прекратил её окончательно. Восстановление становится постоянным, если правообладатель не уведомил о нарушении в течение 60 дней после его прекращения. При первом уведомлении от данного правообладателя постоянное восстановление также наступает, если нарушение устранено в течение 30 дней после получения уведомления. Источник: "+GPL+
  "\n\nApache 2.0 §3: патентная лицензия для соответствующего Work прекращается со дня подачи патентного иска о прямом либо содействующем нарушении патента со стороны Work или включённого вклада, включая встречные и перекрёстные иски. В §3 нет аналогичного GPLv3 §8 механизма восстановления с 30- и 60-дневными сроками. Источник: "+APACHE+
  "\n\nРазличие: GPLv3 §8 прямо описывает восстановление прав после устранения нарушения, а Apache 2.0 §3 — прекращение по указанному патентному иску без аналогичного механизма в тексте §3."};
}
