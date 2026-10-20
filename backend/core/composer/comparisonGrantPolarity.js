import {selectOperativeSourceText} from "../validator/claim/operativeSourceScope.js";

// Never transfer a patent-litigation exception to an expressly irrevocable
// copyright grant. Only repair an affirmative claim contradicted by that
// grant in the retrieved operative primary source.
const GRANT=/\birrevocable\s+copyright\s+license\b/iu;
const REVOCABLE=/(?<![\p{L}])отзываем\p{L}*/iu;
const COPYRIGHT=/авторск\p{L}*\s+лицензи\p{L}*|copyright\s+license/iu;
const EXCEPTION=/\s*\(\s*except\s+in\s+case\s+of\s+(?:patent\s+)?litigation\s*\)/iu;

export function correctCopyrightGrantPolarity(value,document){
 const original=String(value||"");
 const claim=original.split("«",1)[0].split("[",1)[0];
 if(!REVOCABLE.test(claim)||!COPYRIGHT.test(claim))return original;
 const source=selectOperativeSourceText(document).replace(/\s+/g," ");
 if(!GRANT.test(source))return original;
 return original.replace(REVOCABLE,"безотзывную").replace(EXCEPTION,"");
}
