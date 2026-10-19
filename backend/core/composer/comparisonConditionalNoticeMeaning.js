import {selectOperativeSourceText} from "../validator/claim/operativeSourceScope.js";
const FILE_OVERCLAIM=/(сохран\p{L}*|включ\p{L}*|переда\p{L}*)[^.;]{0,110}(?:файл\s+NOTICE|NOTICE\s*[‐‑-]?файл)/iu;
const CONDITION=/\bif the work includes a\b[^.]{0,80}\bnotice\b/iu;
const DUTY=/\bmust include a readable copy\b[^.]{0,210}\battribution notices\b/iu;
const KEEP=/\byou must retain\b[^.]{0,280}\b(?:copyright|attribution)\b[^.]{0,160}\bnotices\b/iu;
export function correctNoticeFileClaim(value,document){
 const original=String(value||"");
 if(!FILE_OVERCLAIM.test(original.replace(/\*\*/g,"")))return original;
 const source=selectOperativeSourceText(document).replace(/\s+/g," ");
 if(!CONDITION.test(source)||!DUTY.test(source))return original;
 const conditional="Если работа содержит файл NOTICE, при распространении производной работы требуется включить читаемую копию атрибуционных уведомлений из этого файла.";
 return KEEP.test(source)?
  "При распространении производных работ сохранять первоначальные уведомления об авторских правах и атрибуции. "+conditional:conditional;
}
