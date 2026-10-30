/** Preserve a source-verified BSD 3-Clause non-endorsement quote as literal data. */
const URL="https://opensource.org/license/bsd-3-clause";
const SENTENCE="Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission."; 

export function isVerifiedBsd3NonendorsementQuote(result,task){
 if(result?.success!==true || result?.validated!==true || result?.mode!=="single")return false;
 const request=String(task||"");
 if(!/\bbsd\s*3[\s-]*clause\b/iu.test(request) ||
    !/(?:английск|english)/iu.test(request) ||
    !/(?:дословн|цитат|verbatim|quote)/iu.test(request) ||
    !/(?:им[её]н|автор|держател|contributor|copyright|endorse|продвижен)/iu.test(request))return false;
 const answer=String(result.text||"");
 if(answer.length>700 || !answer.includes(URL))return false;
 const quotations=[...answer.matchAll(/[«“"]([^»”"\n]*)[»”"]/gu)];
 return quotations.length===1 && quotations[0][1].trim().replace(/^3\.\s*/u,"")===SENTENCE;
}
