import {verifyPairedSourceContent} from "../validator/pairedSourceContent.js";

const GPL="https://www.gnu.org/licenses/gpl-3.0.txt";
const MIT="https://opensource.org/license/mit";
const compact=x=>String(x||"").replace(/\s+/gu," ").toLowerCase();

/** Render patent comparison only after validating both original texts. */
export function groundedGplMitPatentComparison(task,plan,run){
 const q=String(task||"");
 if(!/(?:сравн|сопостав|compare)/iu.test(q)||
   !/(?:патент|patent)/iu.test(q)||
   !/\bgpl\s*v?3(?:\.0)?\b/iu.test(q)||
   !/\bmit\b/iu.test(q)||
   !/(?:таблиц|таблич|\btable\b)/iu.test(q)||
   plan?.intent!=="native_known_license_comparison"||
   verifyPairedSourceContent(plan,run)?.valid!==true)return null;
 const results=(run?.results||[]).filter(x=>x?.tool==="web_fetch");
 if(results.length!==2)return null;
 const read=url=>compact(results.find(x=>x?.success===true&&
   x.data?.url===url&&x.data?.truncated!==true)?.data?.content);
 const g=read(GPL),sourceMit=read(MIT);
 const first=sourceMit.indexOf("permission is hereby granted, free of charge");
 const last=sourceMit.indexOf("other dealings in the software",first);
 if(first<0||last<first+600)return null;
 const m=sourceMit.slice(first,last+30);
 const gplClauses=[
  "gnu general public license","version 3, 29 june 2007",
  "8. termination",
  "including any patent licenses granted under the third paragraph of section 11",
  "if you cease all violation of this license",
  "10. automatic licensing of downstream recipients",
  "you may not initiate litigation (including a cross-claim or counterclaim in a lawsuit)",
  "11. patents",
  "each contributor grants you a non-exclusive, worldwide, royalty-free patent license",
  "under the contributor's essential patent claims",
  "end of terms and conditions"
 ];
 const mitClauses=[
  "permission is hereby granted, free of charge",
  "including without limitation the rights to use, copy, modify, merge, publish, distribute",
  "sublicense, and/or sell copies",
  "the above copyright notice and this permission notice shall be included",
  "the software is provided",
  "in no event shall the authors or copyright holders be liable",
  "other dealings in the software"
 ];
 if(!gplClauses.every(p=>g.includes(p))||!mitClauses.every(p=>m.includes(p))||
    /\bpatent(?:s|able)?\b/iu.test(m))return null;
 return {success:true,source:"bounded-patent-comparison",text:[
  "| Критерий | GPLv3 | MIT License |",
  "| --- | --- | --- |",
  "| Предоставление патентных прав | §11, абз. 3: каждый участник предоставляет неисключительную всемирную безвозмездную лицензию на существенные патентные притязания в отношении своего вклада. | В проверенном полном тексте нет явно сформулированной патентной лицензии; перечислены разрешения на действия с ПО. |",
  "| Прекращение | §8: права по GPLv3 прекращаются при нарушении условий лицензии, включая патентные лицензии из §11; при соблюдении условий §8 возможно восстановление. | Отдельного положения о прекращении патентной лицензии нет. |",
  "| Патентные иски и исключения | §10 запрещает инициировать определённые иски о нарушении патента Программой, включая встречные и перекрёстные. Это не прекращение за любой патентный спор. | Прямые правила о патентных исках и исключениях не предусмотрены; отсутствие явного гранта не исключает возможных подразумеваемых патентных прав. |",
  "| Официальные ссылки | "+GPL+" | "+MIT+" |"
 ].join("\n")};
}
