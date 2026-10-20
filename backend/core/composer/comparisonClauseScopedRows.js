import {selectOperativeSourceText} from "../validator/claim/operativeSourceScope.js";

/*
 * A cell for a specific legal criterion must not become a list of separate
 * duties from other clauses. Only use the narrow, directly quoted operative
 * grant when that exact grant was fetched from the primary source.
 * Do not infer missing permissions, limitations or duties.
 */
const COPYRIGHT_GRANT=/\bhereby grants to you\b[^.]{0,240}\birrevocable copyright license\b[^.]{0,200}\breproduce, prepare derivative works\b/iu;
const REDISTRIBUTION=/\byou may reproduce and distribute copies of the work or derivative works thereof\b[^.]{0,340}\bin source or object form\b[^.]{0,115}\bprovided that you meet the following conditions\b/iu;
const USE_DUTIES=/\b(?:при\s+распространении|требу\p{L}*\s+сохран\p{L}*|требу\p{L}*\s+предостав\p{L}*|помет\p{L}*\s+измен\p{L}*|уведомлен\p{L}*\s+о\s+внес)/iu;
const DISTRIBUTION_DUTIES=/(?:\([abcde]\)|\b(?:при\s+условии|условия)\s*:|\b(?:сохран\p{L}*|помет\p{L}*|включ\p{L}*|предостав\p{L}*)\b[^.;]{0,100}\b(?:уведомлен\p{L}*|лицензи\p{L}*|файл\p{L}*\s+notice)\b)/iu;

export function narrowComparisonRightsClaim(topic,value,document){
 const original=String(value||"");
 const claim=original.split("«",1)[0];
 const source=selectOperativeSourceText(document).replace(/\s+/g," ");
 if(topic==="use"&&USE_DUTIES.test(claim)&&COPYRIGHT_GRANT.test(source))
   return "Предоставляется безотзывная авторская лицензия на воспроизведение и создание производных работ при соблюдении условий лицензии.";
 if(topic==="distribution"&&DISTRIBUTION_DUTIES.test(claim)&&REDISTRIBUTION.test(source))
   return "Разрешается распространять исходную работу и производные работы в исходной или объектной форме, с изменениями или без них, при соблюдении условий лицензии.";
 return original;
}
