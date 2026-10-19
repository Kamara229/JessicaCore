import {selectOperativeSourceText} from "../validator/claim/operativeSourceScope.js";

/* Repair only an overstatement contradicted by the source's operative
 * permission. The underlying original source remains the authority.
 * No skill- or license-name-specific assertions are stored here. */
const ABSOLUTE=/\bбез\s+(?:каких-либо\s+|любых\s+|дополнительных\s+)?ограничен\p{L}*/iu;
const SOURCE_RIGHTS=/\b(?:hereby grants to you|redistribution and use in source and binary forms)\b/iu;
const CONDITIONAL=/\b(?:subject to the terms and conditions|provided that the following conditions|provided that you meet the following conditions)\b/iu;
const REDISTRIBUTION_USE=/\bredistribution and use in source and binary forms\b[^.]{0,210}\bpermitted\b/iu;
export function correctOverstatedUse(value,document) {
  let text=String(value||"");
  const source=selectOperativeSourceText(document).replace(/\s+/g," ");
  if(!SOURCE_RIGHTS.test(source)||!CONDITIONAL.test(source))return text;
  if(ABSOLUTE.test(text)){
    text=text.replace(ABSOLUTE,"при соблюдении условий лицензии")
      .replace(/при соблюдении условий лицензии\s*,?\s*при условии соблюдения условий лицензии/iu,"при соблюдении условий лицензии");
  }
  // Source permission is distinct from downstream redistribution duties.
  // Recast an overloaded 'use' cell only when source explicitly grants
  // both source/binary use and modification (conditional, not absolute).
  if(REDISTRIBUTION_USE.test(source)
    &&/уведомлен|копирайт|copyright|отказ от ответственности/iu.test(text)
    &&/использ|модифиц|изменен/iu.test(text))
      return "Разрешено использование и изменение в исходной и бинарной формах при соблюдении условий лицензии.";
  return text;
}
