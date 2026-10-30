/** Render absolute calendar-day distance with an explicit order-independent calculation. */
export function formatAbsoluteDayExplanation(start,end,days,unit){
  return "Между "+start+" и "+end+" "+days+" "+unit+". "+
    "Расчёт: |"+end+" − "+start+"| = "+days+
    " календарных дней. Берём модуль разницы: порядок дат не влияет.";
}
