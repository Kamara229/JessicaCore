/**
 * Explain the already computed elapsed-day count; don't infer a new count.
 * Enumerate small forward intervals so a user can audit every included day.
 */
export function describeCalendarInterval(start,end,days) {
  const begin=Date.parse(start+"T00:00:00Z");
  const finish=Date.parse(end+"T00:00:00Z");
  if(!Number.isFinite(begin)||!Number.isFinite(finish)||finish<begin)
    return "Расчёт: количество календарных суток между указанными датами.";
  if(days===0)return "Расчёт: даты совпадают, прошедших дней нет.";
  if(days<=10){
    const formatter=new Intl.DateTimeFormat("ru-RU",{
      day:"numeric",month:"long",timeZone:"UTC"
    });
    const included=Array.from({length:days},(_,index)=>
      formatter.format(new Date(begin+(index+1)*86400000))
    );
    return "Расчёт: считаем "+included.join(", ")+" — всего "+days+" дней.";
  }
  return "Расчёт: считаем календарные сутки после "+start+
    " по "+end+" включительно — всего "+days+" дней.";
}
