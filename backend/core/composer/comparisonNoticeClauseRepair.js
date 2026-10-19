import {selectOperativeSourceText} from "../validator/claim/operativeSourceScope.js";

/*
 * Rewrite a false requirement to retain the literal NOTICE file only
 * when the fetched document itself establishes the actual conditional
 * duty to reproduce its attribution contents. Never infer absences.
 */
const FILE_OVERCLAIM=/\b(?:сохран\p{L}*|хран\p{L}*|включ\p{L}*|переда\p{L}*)\b[^.;]{0,95}\b(?:файл\s+NOTICE|NOTICE\s*[‐‑-]?файл|\bNOTICE\b)/iu;
const CONDITION=/\bif the work includes a\b[^.]{0,75}\bnotice\b/iu;
const NOTICE_COPY=/\bmust include a readable copy\b[^.]{0,190}\b(?:attribution notices|notices contained)\b/iu;
const RETENTION=/\byou must retain\b[^.]{0,235}\b(?:copyright|patent|trademark|attribution)\b[^.]{0,135}\bnotices\b/iu;
export function correctNoticeFileMeaning(value,document) {
  const input=String(value||"");
  if(!FILE_OVERCLAIM.test(input))return input;
  const source=selectOperativeSourceText(document).replace(/\s+/g," ");
  if(!CONDITION.test(source)||!NOTICE_COPY.test(source))return input;
  const terms=RETENTION.test(source);
  const conditional="Если исходная работа содержит файл NOTICE, при распространении производной работы необходимо включить читаемую копию содержащихся в нём атрибуционных уведомлений.";
  return terms?
    "При распространении производных работ сохранять исходные уведомления об авторских правах, патентах, товарных знаках и атрибуции в их исходной форме. "+conditional:
    conditional;
}
