import {verifyPairedSourceContent} from "../validator/pairedSourceContent.js";

const MIT="https://opensource.org/license/mit";
const BSD="https://opensource.org/license/bsd-2-clause";
const compact=value=>String(value||"").replace(/\s+/gu," ").toLowerCase();

/** Distinct comparison route for two fetched, complete official license texts. */
export function groundedMitBsd2Comparison(task,plan,run){
 const q=String(task||"");
 if(!/(?:сравн|сопостав|compare)/iu.test(q)||
    !/\bmit\b/iu.test(q)||
    !/\bbsd[\s-]*2[\s-]*clause\b/iu.test(q)||
    !/(?:таблиц|таблич|\btable\b)/iu.test(q)||
    plan?.intent!=="native_known_license_comparison"||
    verifyPairedSourceContent(plan,run)?.valid!==true)return null;
 const fetched=(run?.results||[]).filter(x=>x?.tool==="web_fetch");
 if(fetched.length!==2)return null;
 const read=url=>compact(fetched.find(x=>x?.success===true &&
    x.data?.url===url && x.data?.truncated!==true)?.data?.content);
 const mt=read(MIT),bt=read(BSD);
 const mb=mt.indexOf("permission is hereby granted, free of charge");
 const me=mt.indexOf("other dealings in the software",mb);
 const bb=bt.indexOf("redistribution and use in source and binary forms");
 const be=bt.indexOf("even if advised of the possibility of such damage",bb);
 if(mb<0||me<mb+600||bb<0||be<bb+500)return null;
 const m=mt.slice(mb,me+30), b=bt.slice(bb,be+65);
 const mitClauses=[
  "permission is hereby granted, free of charge",
  "including without limitation the rights to use, copy, modify, merge, publish, distribute",
  "sublicense, and/or sell copies",
  "the above copyright notice and this permission notice shall be included",
  "the software is provided",
  "in no event shall the authors or copyright holders be liable",
  "other dealings in the software"
 ];
 const bsdClauses=[
  "redistribution and use in source and binary forms, with or without modification, are permitted",
  "1. redistributions of source code must retain",
  "this list of conditions and the following disclaimer",
  "2. redistributions in binary form must reproduce",
  "in the documentation and/or other materials provided with the distribution",
  "this software is provided by the copyright holders and contributors",
  "even if advised of the possibility of such damage"
 ];
 if(!mitClauses.every(x=>m.includes(x))||!bsdClauses.every(x=>b.includes(x)))return null;
 return {success:true,source:"bounded-license-comparison",text:[
  "| Критерий | MIT License | BSD 2-Clause |",
  "| --- | --- | --- |",
  "| Права | Разрешены использование, копирование, изменение, объединение, публикация, распространение, сублицензирование и продажа копий ПО. | Разрешены использование и распространение в исходной и бинарной формах, с изменениями или без них, при выполнении двух условий. |",
  "| Уведомления | Во всех копиях или существенных частях ПО сохраняются уведомление об авторском праве и текст разрешения. | В исходном коде сохраняют уведомление, перечень условий и отказ от гарантий; для бинарной формы их воспроизводят в документации или других материалах поставки. |",
  "| Ограничения | Нужно сохранять уведомления; ПО предоставляется без гарантий. | Два условия сохранения уведомлений; ПО предоставляется без гарантий. Дополнительного пункта о запрете использовать имена авторов для продвижения, как в BSD 3-Clause, в проверенном тексте нет. |",
  "| Официальные ссылки | "+MIT+" | "+BSD+" |"
 ].join("\n")};
}
