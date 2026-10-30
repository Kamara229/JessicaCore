const GPL3_URL="https://www.gnu.org/licenses/gpl-3.0.txt";
const GPL3_QUOTE="conspicuously and appropriately publish on each copy an appropriate copyright notice";

/** Keep the exact English quotation for the explicitly requested primary GNU text. */
function isVerifiedGplSection4Quote(result,task){
 if(result?.success!==true || result?.validated!==true ||
    result?.answerSource!=="bounded-gpl-section4-excerpt")return false;
 const request=String(task||"");
 if(!/(?:§\s*4|section\s*4|раздел\s*4)/iu.test(request) ||
    !/\bgpl\s*(?:v\s*)?3(?:\.0)?\b/iu.test(request) ||
    !/(?:дословн|цитат|verbatim|quote)/iu.test(request) ||
    !/(?:уведомлен|авторск|copyright|notice)/iu.test(request))return false;
 const answer=String(result?.text||"");
 const expected="§4 GPLv3: “"+GPL3_QUOTE+"”.\nОфициальный URL: "+GPL3_URL;
 return answer===expected;
}
/**
 * An English-language source excerpt is data, not an English explanation.
 * The exception is limited to fully validated single-task source excerpts.
 */
export function isVerifiedRequestedOriginalQuote(result,task){
    if(isVerifiedGplSection4Quote(result,task))return true;
    if(!result || result.success!==true || result.validated!==true ||
       result.answerSource!=="bounded-original-license-excerpt")return false;
    const request=String(task||"");
    if(!/(?:английск|english)/iu.test(request) ||
       !/(?:патент|patent)/iu.test(request) ||
       !/(?:§\s*3|раздел\s*3|section\s*3)/iu.test(request))return false;
    const content=String(result.text||"");
    if(!content.includes("https://www.apache.org/licenses/LICENSE-2.0.txt"))return false;
    const quote=/[«“]([^»”\n]{25,230})[»”]/u.exec(content)?.[1]||"";
    const words=quote.trim().split(/\s+/u);
    return words.length===12 &&
        /^(?:licenses granted to You under this License for that Work shall terminate)$/u.test(quote.trim());
}
