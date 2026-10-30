/** Extract MIT notice only from a freshly fetched and validated primary text. */
const OSI="https://opensource.org/license/mit";
const RAW="https://raw.githubusercontent.com/spdx/license-list-data/main/text/MIT.txt";
const SPDX="https://spdx.org/licenses/MIT.html";
const QUOTE="The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.";

export function isMitCopyrightNoticeRequest(task,plan) {
 const request=String(task||"");
 return plan?.intent==="native_single_license_primary_text" &&
   plan?.metadata?.licenseId==="MIT" &&
   /\bmit\b/iu.test(request) &&
   /(?:дословн|цитат|verbatim|quote)/iu.test(request) &&
   /(?:copyright|авторск|уведомлен|notice)/iu.test(request);
}

export function extractMitCopyrightNotice(task,plan,run) {
 if(!isMitCopyrightNoticeRequest(task,plan))return null;
 const steps=Array.isArray(plan?.steps)?plan.steps:[];
 const rows=Array.isArray(run?.results)?run.results:[];
 if(steps.length!==1 || steps[0]?.tool!=="web_fetch" ||
   steps[0]?.arguments?.url!==OSI || rows.length!==1 ||
   rows[0]?.tool!=="web_fetch" || rows[0]?.success!==true ||
   rows[0]?.data?.truncated!==false)return null;
 const data=rows[0].data||{};
 const fromOsi=data.url===OSI;
 const fromSpdx=data.url===RAW && data.referenceUrl===SPDX;
 if(!fromOsi && !fromSpdx)return null;
 const raw=String(data.content||"");
 if(raw.length<750 || raw.length>20000)return null;
 const normalized=raw.replace(/\s+/gu," ");
 const low=normalized.toLowerCase();
 const required=["mit license","permission is hereby granted, free of charge",
  "including without limitation the rights to use, copy, modify, merge, publish, distribute",
  "the software is provided","in no event shall the authors or copyright holders be liable",
  "other dealings in the software"];
 if(!required.every(x=>low.includes(x)) || !normalized.includes(QUOTE))return null;
 const link=fromOsi?OSI:SPDX;
 return {success:true,source:"bounded-mit-notice-excerpt",
  text:"MIT License: “"+QUOTE+"”.\nОфициальный URL: "+link};
}
