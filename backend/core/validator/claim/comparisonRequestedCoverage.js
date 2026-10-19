/* The requested criteria must appear in the table; this does not
 * certify the meaning of any license provision. */
const TOPICS=[
  {request:/использов|модифиц|изменен|\buse\b|modif/iu,
   row:/использов|модифиц|изменен|\buse\b|modif/iu},
  {request:/распростран|передач|дистрибуц|distribut/iu,
   row:/распростран|передач|дистрибуц|distribut/iu},
  {request:/уведомлен|авторств|копирайт|copyright|notic|attribut/iu,
   row:/уведомлен|авторств|копирайт|copyright|notic|attribut/iu},
  {request:/патент|patent/iu,row:/патент|patent/iu},
  {request:/огранич|товарн|бренд|trademark|restrict/iu,
   row:/огранич|товарн|бренд|trademark|restrict/iu}
];
export function checkRequestedComparisonCoverage(task,matrix) {
  if(!matrix?.criteria?.length)return {valid:false,reason:"Сравнительная таблица отсутствует"};
  const missing=TOPICS.filter(topic=>topic.request.test(String(task||"")) &&
    !matrix.criteria.some(row=>topic.row.test(row)));
  if(missing.length)return {valid:false,reason:"Не раскрыто запрошенных критериев: "+missing.length};
  if(!matrix.checks.some(c=>c.source===0)||!matrix.checks.some(c=>c.source===1))
    return {valid:false,reason:"Нет проверяемых фактов по одному из двух источников"};
  return {valid:true,reason:"Все запрошенные критерии представлены в таблице"};
}
