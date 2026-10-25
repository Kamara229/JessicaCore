/**
 * An English-language source excerpt is data, not an English explanation.
 * The exception is limited to fully validated single-task source excerpts.
 */
export function isVerifiedRequestedOriginalQuote(result,task){
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
