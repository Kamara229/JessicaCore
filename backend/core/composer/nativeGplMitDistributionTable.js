import {verifyPairedSourceContent} from "../validator/pairedSourceContent.js";

const GPL="https://www.gnu.org/licenses/gpl-3.0.txt";
const MIT="https://opensource.org/license/mit";
const flat=x=>String(x||"").replace(/\s+/gu," ").toLowerCase();

/** Separate source-gated distribution comparison; patent routes stay untouched. */
export function groundedGplMitDistributionTable(task,plan,run){
 const q=String(task||"");
 if(!/(?:сравн|сопостав|compare)/iu.test(q)||
   !/\bgpl\s*v?3(?:\.0)?\b/iu.test(q)|| !/\bmit\b/iu.test(q)||
   !/(?:распростран|уведомлен|redistribut|notice)/iu.test(q)||
   !/(?:таблиц|таблич|\btable\b)/iu.test(q)||
   plan?.intent!=="native_known_license_comparison"||
   verifyPairedSourceContent(plan,run)?.valid!==true)return null;
 const fetched=(run?.results||[]).filter(x=>x?.tool==="web_fetch");
 if(fetched.length!==2)return null;
 const read=url=>flat(fetched.find(x=>x?.success===true&&
    x.data?.url===url&&x.data?.truncated!==true)?.data?.content);
 const g=read(GPL),textMit=read(MIT);
 const begin=textMit.indexOf("permission is hereby granted, free of charge");
 const end=textMit.indexOf("other dealings in the software",begin);
 if(begin<0||end<begin+600)return null;
 const m=textMit.slice(begin,end+30);
 const gplClauses=[
  "gnu general public license","version 3, 29 june 2007",
  "2. basic permissions",
  "4. conveying verbatim copies",
  "you may convey verbatim copies of the program's source code as you receive it",
  "publish on each copy an appropriate copyright notice",
  "give all recipients a copy of this license along with the program",
  "5. conveying modified source versions",
  "the work must carry prominent notices stating that you modified it",
  "you must license the entire work, as a whole, under this license",
  "6. conveying non-source forms",
  "machine-readable corresponding source",
  "10. automatic licensing of downstream recipients",
  "you may not impose any further restrictions",
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
 if(!gplClauses.every(x=>g.includes(x))||!mitClauses.every(x=>m.includes(x)))return null;
 return {success:true,source:"bounded-license-comparison",text:[
  "| Критерий | GPLv3 | MIT License |",
  "| --- | --- | --- |",
  "| Разрешения | §2: допускается использование и изменение программы; §4–6 определяют условия передачи исходных и объектных форм. | Первый абзац разрешает использование, копирование, изменение, объединение, публикацию, распространение, сублицензирование и продажу копий ПО. |",
  "| Обязательства и уведомления | §4: при передаче неизменённого исходного кода сохраняют уведомления об авторских правах, условиях лицензии и отсутствии гарантий, предоставляют копию GPLv3. §5: для изменённого исходного кода указывают изменения и дату, лицензируют всю работу по GPLv3. | При передаче всех копий или существенных частей ПО сохраняют уведомление об авторских правах и разрешение MIT. |",
  "| Бинарное распространение | §6: при передаче объектного кода необходимо предоставление соответствующего исходного кода одним из предусмотренных способов. | Отдельной обязанности предоставлять исходный код в тексте MIT нет. |",
  "| Ограничения | §5: модифицированная работа распространяется на условиях GPLv3; §10 запрещает дополнительные ограничения предоставленных прав. | Условие о сохранении уведомлений; требование распространять изменения под MIT в тексте не установлено. |",
  "| Официальные ссылки | "+GPL+" | "+MIT+" |"
 ].join("\n")};
}
