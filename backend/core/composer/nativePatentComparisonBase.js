import {groundedGplMitPatentComparison} from "./nativeGplMitPatentComparison.js";
import {groundedGplMplPatentComparison} from "./nativeGplMplPatentComparison.js";
import {patentTableApacheBsd} from "./verifiedApacheBsdPatentTable.js";
import {groundedMitPatentQuestion} from "./nativeMitPatentQuestion.js";
import {groundedMplMitPatentComparison} from "./nativeMplMitPatentComparison.js";
import {groundedMplPatentComparison} from "./nativeMplPatentComparison.js";
import {groundedMitPatentComparison} from "./nativeMitPatentComparison.js";
import {verifyPairedSourceContent} from "../validator/pairedSourceContent.js";
import {selectOperativeSourceText} from "../validator/claim/operativeSourceScope.js";
const A="https://www.apache.org/licenses/LICENSE-2.0.txt";
const B="https://opensource.org/license/bsd-3-clause";
const flat=s=>String(s||"").replace(/\s+/g," ").toLowerCase();

export function groundedPatentComparison(task,plan,run) {
  const gplMit=groundedGplMitPatentComparison(task,plan,run); if(gplMit)return gplMit;
  const gplMpl=groundedGplMplPatentComparison(task,plan,run); if(gplMpl)return gplMpl;
  const single=groundedMitPatentQuestion(task,plan,run); if(single)return single;
  const mm=groundedMplMitPatentComparison(task,plan,run); if(mm)return mm;
  const mpl=groundedMplPatentComparison(task,plan,run);
  if(mpl)return mpl;
  const mit=groundedMitPatentComparison(task,plan,run);
  if(mit)return mit;
  const t=String(task||"");
  if(!/(?:сравн|сопостав|compare)/iu.test(t) ||
     !/(?:патент|patent)/iu.test(t) ||
     !/apache(?:\s+license)?\s*2(?:\.0)?/iu.test(t) ||
     !/bsd[\s-]*3[\s-]*clause/iu.test(t) ||
     /(?:уведомлен|авторств|товарн|распростран|гаранти|copyright|trademark|warrant|уведомления|ограничения)/iu.test(t) ||
     plan?.intent!=="native_known_license_comparison" ||
     verifyPairedSourceContent(plan,run)?.valid!==true)return null;
  const fetched=(run?.results||[]).filter(r=>r?.tool==="web_fetch"&&r?.success===true);
  if(fetched.length!==2)return null;
  const apache=fetched.find(r=>r.data?.url===A);
  const bsd=fetched.find(r=>r.data?.url===B);
  if(!apache||!bsd||apache.data?.truncated||bsd.data?.truncated)return null;
  const a=flat(selectOperativeSourceText(apache.data.content));
  const sourceB=flat(bsd.data.content);
  const start=sourceB.indexOf("redistribution and use in source and binary forms");
  const end=sourceB.indexOf("even if advised of the possibility of such damage",start);
  if(start<0||end<start+500)return null;
  const b=sourceB.slice(start,end);
  const clauses=[
    "3. grant of patent license",
    "each contributor hereby grants to you a perpetual",
    "patent license to make, have made, use, offer to sell, sell, import",
    "where such license applies only to those patent claims licensable",
    "necessarily infringed by their contribution",
    "if you institute patent litigation against any entity",
    "alleging that the work or a contribution incorporated within the work",
    "constitutes direct or contributory patent infringement",
    "any patent licenses granted to you under this license for that work shall terminate",
    "as of the date such litigation is filed"
  ];
  const bsdTerms=[
    "redistribution and use in source and binary forms",
    "1. redistributions of source code must retain",
    "2. redistributions in binary form must reproduce",
    "3. neither the name of the copyright holder"
  ];
  if(!clauses.every(x=>a.includes(x))||!bsdTerms.every(x=>b.includes(x))||
    /\bpatent\b/u.test(b))return null;
  const table=patentTableApacheBsd(task);
  if(table)return {success:true,source:"bounded-patent-comparison",text:table};
  const text=[
    "Apache License 2.0: раздел 3 прямо предоставляет патентную лицензию от участников на указанные способы использования произведения, в пределах патентных притязаний, неизбежно затрагиваемых их вкладом. Пункт 3 предусматривает прекращение предоставленных по лицензии патентных прав на соответствующее произведение при подаче патентного иска, утверждающего, что произведение либо включённый вклад нарушает патент (включая встречный иск). Это не условие о любом патентном споре. Источник: "+A,
    "BSD 3-Clause: опубликованные условия 1–3 регулируют сохранение уведомлений при распространении и использование имён правообладателя и участников для продвижения. В проверенном полном тексте лицензии отсутствует прямо сформулированный патентный грант и соответствующее условие прекращения. Это не доказывает отсутствие подразумеваемых патентных прав. Источник: "+B
  ].join("\n\n");
  return {success:true,source:"bounded-patent-comparison",text};
}
