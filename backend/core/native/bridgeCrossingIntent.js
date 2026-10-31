/** Recognize only complete, explicitly quantified 4–6 person bridge puzzles. */
export function parseBridgeCrossingTask(task) {
  const t=String(task||"");
  if(!t || t.length>1600 ||
     !/(мост|bridge)/iu.test(t) ||
     !/(фонар|torch|flashlight)/iu.test(t) ||
     !/(двое|двоих|двух|two\s+at\s+a\s+time|максимум\s*2|не\s+более\s*2)/iu.test(t))return null;

  const counts=[
    /(?:четверо|4\s*(?:человека|человек|люд))/iu.test(t)?4:null,
    /(?:пятеро|5\s*(?:человек|люд))/iu.test(t)?5:null,
    /(?:шестеро|6\s*(?:человек|люд))/iu.test(t)?6:null
  ].filter(Boolean);
  if(counts.length!==1)return null;

  const m=t.match(/(?:за|время\s*:|времена\s*:?)\s*((?:\d{1,3}\s*(?:,|;|и|and)\s*){3,5}\d{1,3})\s*(?:мин|minutes?)/iu);
  if(!m)return null;
  const times=m[1].match(/\d+/gu)?.map(Number)||[];
  if(times.length!==counts[0] || times.some(n=>!Number.isSafeInteger(n)||n<1||n>120))
    return null;
  return {times,explain:/(докаж|объясн|почему|обосну|строг|вариант|оптимальн|маршрут|prove|why|explain)/iu.test(t)};
}
