/** Exact four-person bridge task: only parse explicitly supplied times. */
export function parseBridgeCrossingTask(task) {
  const t=String(task||"");
  if(t.length>1600 ||
     !/(мост|bridge)/iu.test(t) ||
     !/(фонар|torch|flashlight)/iu.test(t) ||
     !/(четверо|4\s*(?:человек|люд)|four\s+people)/iu.test(t) ||
     !/(двое|двоих|двух|two\s+at\s+a\s+time|максимум\s*2)/iu.test(t)) return null;
  const m=t.match(/(?:за|времена\s*:?)\s*(\d{1,3}(?:\s*(?:,|;|и|and)\s*\d{1,3}){3})\s*(?:мин|minutes)/iu);
  if(!m)return null;
  const times=m[1].match(/\d+/gu)?.map(Number)||[];
  if(times.length!==4 || times.some(n=>!Number.isSafeInteger(n)||n<1||n>120)
     || new Set(times).size!==4) return null;
  return {times,explain:/(докаж|объясн|почему|обосну|строг|вариант|prove|why|explain)/iu.test(t)};
}
