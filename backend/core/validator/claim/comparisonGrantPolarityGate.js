// An independently checked quote may have valid provenance and still
// contradict a claimed legal effect. A positive source statement of
// copyright irrevocability contradicts an asserted revocable copyright
// grant, even when patent-litigation language appears elsewhere.
const REVOCABLE=/(?<![\p{L}])отзываем\p{L}*/iu;
const COPYRIGHT=/авторск\p{L}*\s+лицензи\p{L}*|copyright\s+license/iu;
const IRREVOCABLE=/\birrevocable\s+copyright\s+license\b|\bcopyright\s+license[^.]{0,80}\birrevocable\b/iu;

export function contradictsExpressCopyrightGrant(cell,quotes){
 const statement=String(cell?.value||"").split("«",1)[0];
 if(!REVOCABLE.test(statement)||!COPYRIGHT.test(statement))return false;
 return IRREVOCABLE.test((quotes||[]).join(" "));
}
