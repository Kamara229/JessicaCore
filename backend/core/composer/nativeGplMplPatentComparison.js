import {verifyPairedSourceContent} from "../validator/pairedSourceContent.js";
const GPL="https://www.gnu.org/licenses/gpl-3.0.txt";
const MPL="https://www.mozilla.org/media/MPL/2.0/index.txt";
const compact=x=>String(x||"").replace(/\s+/g," ").toLowerCase();
export function groundedGplMplPatentComparison(task,plan,run) {
  const q=String(task||"");
  if(!/(сравн|сопостав|compare)/iu.test(q)||!/(патент|patent)/iu.test(q)||
     !/\bgpl\s*v?3(?:\.0)?\b/iu.test(q)||!/\bmpl\s*2(?:\.0)?\b/iu.test(q)||
     plan?.intent!=="native_known_license_comparison"||
     verifyPairedSourceContent(plan,run)?.valid!==true)return null;
  const items=(run?.results||[]).filter(x=>x.tool==="web_fetch");
  if(items.length!==2||items.some(x=>x.success!==true||x.data?.truncated===true))return null;
  const read=url=>compact(items.find(x=>x.data?.url===url)?.data?.content);
  const g=read(GPL),m=read(MPL);
  const gpl=[
    "gnu general public license","version 3, 29 june 2007",
    "8. termination","including any patent licenses granted under the third paragraph of section 11",
    "if you cease all violation of this license",
    "10. automatic licensing of downstream recipients",
    "you may not initiate litigation (including a cross-claim or counterclaim in a lawsuit)",
    "11. patents","each contributor grants you a non-exclusive, worldwide, royalty-free patent license",
    "under the contributor's essential patent claims","end of terms and conditions"
  ];
  const mpl=["mozilla public license","2.1. grants","under patent claims of such contributor",
    "2.3. limitations on grant scope",
    "5.1. the rights granted under this license will terminate automatically",
    "5.2. if you initiate litigation against any entity by asserting a patent infringement claim",
    "excluding declaratory judgment actions, counter-claims, and cross-claims",
    "under section 2.1 of this license shall terminate"];
  if(!gpl.every(x=>g.includes(x))||!mpl.every(x=>m.includes(x)))return null;
  return {success:true,source:"bounded-patent-comparison",text:
    "GPLv3: раздел 11, абзац 3 предоставляет от каждого участника неисключительную всемирную безвозмездную лицензию на его существенные патентные притязания. Раздел 8 прекращает права при распространении или изменении программы вопреки GPLv3 (включая патентные лицензии третьего абзаца раздела 11) и предусматривает восстановление при условиях, указанных в разделе 8. Раздел 10 запрещает патентные иски по программе, в том числе встречные и перекрёстные. Это не прекращение за любой патентный спор. Источник: "+GPL+
    "\n\nMPL 2.0: пункт 2.1(b) предоставляет патентную лицензию на вклад участника или Contributor Version; пункт 2.3 ограничивает объём. Пункт 5.1 прекращает права при несоблюдении лицензии и предусматривает восстановление. Пункт 5.2 устанавливает отдельное прекращение прав по пункту 2.1 при определённом патентном иске; иски о признании права, встречные и перекрёстные иски исключены. Источник: "+MPL+
    "\n\nГлавное различие: GPLv3 соединяет общий механизм прекращения (§8) с запретом патентных исков (§10), а MPL 2.0 устанавливает отдельное условие по патентному иску (§5.2) с явными исключениями."};
}
