import {selectOperativeSourceText} from "../validator/claim/operativeSourceScope.js";

// Never transfer a patent-litigation exception to an expressly irrevocable
// copyright grant. Only repair an affirmative claim contradicted by that
// grant in the retrieved operative primary source.
const GRANT=/\birrevocable\s+copyright\s+license\b/iu;
const REVOCABLE=/(?<![\p{L}])отзываем\p{L}*/iu;
const COPYRIGHT=/авторск\p{L}*\s+лицензи\p{L}*|copyright\s+license/iu;
const EXCEPTION=/\s*\(\s*except\s+in\s+case\s+of\s+(?:patent\s+)?litigation\s*\)/iu;
const ENDINGS={
 "ая":"безотзывная","ую":"безотзывную","ой":"безотзывной",
 "ый":"безотзывный","ое":"безотзывное","ые":"безотзывные",
 "ых":"безотзывных","ым":"безотзывным","ыми":"безотзывными",
 "ого":"безотзывного","ому":"безотзывному","ом":"безотзывном"
};
function inflectIrrevocable(match){
 const resolved=ENDINGS[match.toLowerCase().slice("отзываем".length)];
 if(!resolved)return match;
 return match[0]===match[0].toUpperCase()
   ?resolved[0].toUpperCase()+resolved.slice(1):resolved;
}

export function correctCopyrightGrantPolarity(value,document){
 const original=String(value||"");
 const claim=original.split("«",1)[0].split("[",1)[0];
 if(!REVOCABLE.test(claim)||!COPYRIGHT.test(claim))return original;
 const source=selectOperativeSourceText(document).replace(/\s+/g," ");
 if(!GRANT.test(source))return original;
 const corrected=original.replace(REVOCABLE,inflectIrrevocable);
 if(corrected===original)return original;
 return corrected.replace(EXCEPTION,"").replace(/(?<![\p{L}])об\s+безотзывной/iu,"о безотзывной");
}
