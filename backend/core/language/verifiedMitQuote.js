/** Preserve exactly source-verified MIT notice through UI localization. */
const PHRASE="The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.";
const OSI="https://opensource.org/license/mit";
const SPDX="https://spdx.org/licenses/MIT.html";
export function isVerifiedMitOriginalQuote(result,task){
 if(result?.success!==true || result?.validated!==true ||
    result?.mode!=="single" || result?.answerSource!=="bounded-mit-notice-excerpt")return false;
 const q=String(task||"");
 if(!/\bmit\b/iu.test(q) || !/(?:английск|english)/iu.test(q) ||
    !/(?:дословн|цитат|verbatim|quote)/iu.test(q) ||
    !/(?:copyright|авторск|уведомлен|notice)/iu.test(q))return false;
 const text=String(result.text||"");
 return text==="MIT License: “"+PHRASE+"”.\nОфициальный URL: "+OSI ||
        text==="MIT License: “"+PHRASE+"”.\nОфициальный URL: "+SPDX;
}
