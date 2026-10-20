/*
 * A limitation on trademark use does not, by itself, prove a separate
 * prohibition on claiming official approval/endorsement. Only cited
 * operative terms describing that second restriction can support it.
 */
const ASSERTS_APPROVAL=/(?:запрещ\p{L}*|нельзя|не\s+разреш\p{L}*)[^.;]{0,130}(?:заявля\p{L}*|утвержда\p{L}*|указыв\p{L}*)[^.;]{0,130}(?:одобр\p{L}*|поддерж\p{L}*|рекоменд\p{L}*)/iu;
const EXPLICIT_APPROVAL=/\b(?:endorse|endorsed|endorsement|promote|promotion|approved|approval|sponsor|sponsorship)\b/iu;

export function unsupportedEndorsementClaim(value,quotes){
  const statement=String(value||"").split("«",1)[0];
  if(!ASSERTS_APPROVAL.test(statement))return false;
  return !EXPLICIT_APPROVAL.test((Array.isArray(quotes)?quotes:[]).join(" "));
}
