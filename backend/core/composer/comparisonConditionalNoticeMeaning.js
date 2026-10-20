import {selectOperativeSourceText} from "../validator/claim/operativeSourceScope.js";

// Only narrow a claim when the fetched operative clauses prove both the
// conditional NOTICE attribution duty and the relevant retention duty.
// A claim about the whole file or "all notices" must not replace their scope.
const FILE=/\bNOTICE\b/iu;
const FILE_OVERCLAIM=/(?:сохран\p{L}*|включ\p{L}*|переда\p{L}*|хран\p{L}*)[^.;]{0,120}(?:файл\s+NOTICE|NOTICE\s*[‐‑-]?файл)/iu;
const WHOLE_CONTENT=/(?:включ\p{L}*|переда\p{L}*|сохран\p{L}*)[^.;]{0,95}(?:его\s+)?(?:содержим\p{L}*|текст\p{L}*)/iu;
const UNQUALIFIED_NOTICES=/(?:все|любые)\s+(?:(?:первоначальн\p{L}*|исходн\p{L}*)\s+)?уведомлен\p{L}*/iu;
const CONDITION=/\bif the work includes a\b[^.]{0,80}\bnotice\b/iu;
const DUTY=/\bmust include a readable copy\b[^.]{0,210}\battribution notices\b/iu;
const KEEP=/\byou must retain\b[^.]{0,280}\b(?:copyright|attribution)\b[^.]{0,160}\bnotices\b/iu;
export function correctNoticeFileClaim(value,document) {
  const original=String(value||"");
  // Ignore Markdown styling, nonbreaking spaces and explanatory quotations.
  // What matters is the *asserted duty*, not a misleading cited excerpt.
  const clean=original.replace(/[\u00a0\u202f]/g," ").replace(/[\\*_`]/g,"");
  const assertion=clean.split("«",1)[0];
  const ACTION=/(?:сохран\p{L}*|включ\p{L}*|переда\p{L}*|хран\p{L}*|воспроизв\p{L}*|копир\p{L}*)/iu;
  const FILE_OR_CONTENT=/(?:файл\p{L}*\s+NOTICE|NOTICE\s*[‐‑-]?файл\p{L}*|содерж(?:им|ан)\p{L}*|текст\p{L}*)/iu;
  // Avoid rewriting already-qualified or explicitly denied requirements.
  if(/без\s+(?:обязательств|требований)|не\s+(?:обязывает|требует|нужно)|нет\s+обязанности/iu.test(assertion))return original;
  if(/(?:если|когда)/iu.test(assertion) && /читаем\p{L}*\s+копи\p{L}*/iu.test(assertion) && /атрибуц\p{L}*\s+уведомлен\p{L}*/iu.test(assertion))return original;
  const assertedWholeFile=FILE.test(assertion)&&ACTION.test(assertion)&&FILE_OR_CONTENT.test(assertion);
  if(!FILE.test(assertion)||
     !(assertedWholeFile||FILE_OVERCLAIM.test(assertion)||
       WHOLE_CONTENT.test(assertion)||UNQUALIFIED_NOTICES.test(assertion)))return original;
  const source=selectOperativeSourceText(document).replace(/\s+/g," ");
  if(!CONDITION.test(source)||!DUTY.test(source))return original;
  const notice="Если исходная работа содержит файл NOTICE, при распространении производной работы требуется включить читаемую копию относящихся к ней атрибуционных уведомлений из этого файла.";
  if(!KEEP.test(source))return notice;
  return "При распространении производных работ в их исходной форме требуется сохранять относящиеся к ним уведомления об авторских правах, патентах, товарных знаках и атрибуции. "+notice;
}
