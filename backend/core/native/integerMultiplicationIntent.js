// Strict, bounded extraction of one explicit integer product from a user task.
// Do not turn general mathematics, multiple operations or code into a native plan.
const REQUEST = /^\s*(?:вычисли|рассчитай|посчитай|сколько будет|calculate|multiply)\s*:?\s*/iu;
const PRODUCT = /^(\(?[+-]?\d{1,15}\)?)\s*[×*xх]\s*(\(?[+-]?\d{1,15}\)?)/iu;

function operand(value) {
  const clean=String(value).replace(/[()\s]/gu,"");
  return /^[+-]?\d{1,15}$/u.test(clean) ? clean : null;
}

export function parseIntegerProductTask(task) {
  if (typeof task!=="string" || task.length>500) return null;
  const request=REQUEST.exec(task);
  if (!request) return null;
  const text=task.slice(request[0].length);
  const match=PRODUCT.exec(text);
  if (!match) return null;
  const a=operand(match[1]), b=operand(match[2]);
  if (a===null || b===null) return null;
  const tail=text.slice(match[0].length).trim();
  // The suffix may ask for an explanation, but not another calculation.
  if (/[\d+×*xх/=]/iu.test(tail)) return null;
  if (tail && !/^[\s.!?,:;а-яёa-z-]+$/iu.test(tail)) return null;
  return {a,b,showWork:/(покажи|расч[её]т|объясн|разлож|по разряд|show|explain|steps)/iu.test(tail)};
}
