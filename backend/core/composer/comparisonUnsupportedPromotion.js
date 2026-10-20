import {selectOperativeSourceText} from "../validator/claim/operativeSourceScope.js";
const HAS_APPROVAL=/\b(?:endorse|endorsed|approval|approved|promote|promotion)\b/iu;
const SEPARATE_PROHIBITION=/;\s*запрещ\p{L}*[^.;]{0,110}заявля\p{L}*[^.;]{0,120}одобр\p{L}*[^.]*\./iu;
export function removeUnsupportedPromotion(value,document){
 const valueText=String(value||"");
 if(HAS_APPROVAL.test(selectOperativeSourceText(document)))return valueText;
 return valueText.replace(SEPARATE_PROHIBITION,".");
}
