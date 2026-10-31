// Check quoted output labels against a finite explicitly enumerated task label set.
// It is a narrow *presentation* guard, not a substitute for semantic reasoning.
const words=text=>String(text||"").toLocaleLowerCase("ru").match(/\p{L}{3,}/gu)||[];
const stop=new Set(["или","либо","только","это","есть","при","для","и","или","без","все","лишь","что","где","надпись","ярлык","одного"]);
export function undeclaredQuotedLabel(task,answer){
  const src=String(task||"");
  // User-defined short label list: e.g. "Категории: A, B, C.".
  const match=src.match(/(?:^|[.!?]\s+)[^.!?\n:]{1,75}:\s*([^.!?\n:]{3,200})\./u);
  if(!match)return null;
  const labels=match[1].split(",").map(s=>s.trim()).filter(Boolean);
  if(labels.length<3||labels.length>10||labels.some(s=>s.length>50))return null;
  const declared=new Set(labels.flatMap(words));
  const taskWords=new Set(words(src));
  const quoted=[...String(answer||"").matchAll(/[«“"]([^»”"\n]{2,100})[»”"]/gu)];
  for(const item of quoted){
    const normalized=words(item[1]).filter(s=>!stop.has(s));
    if(!normalized.some(w=>declared.has(w)))continue;
    const unknown=normalized.find(w=>!taskWords.has(w));
    if(unknown)return {valid:false,shouldRetry:false,
      reason:"В ответе изменены исходные обозначения из задачи: «"+item[1]+"». Используй только названия, указанные пользователем."};
  }
  return null;
}
