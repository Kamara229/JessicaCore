/** A zero-length half-open calendar interval has no counted dates. */
export function formatZeroElapsedDays(start,end,days,mode) {
  if(mode!=="elapsed" || days!==0 || start!==end) return null;
  return "От "+start+" до "+end+" прошло 0 полных календарных дней. "+
    "Начальная дата "+start+" исключается, конечная граница "+end+
    " включается, но даты совпадают. Расчёт: ("+start+", "+end+
    "] — пустой интервал; учитываемых календарных дней нет, всего 0.";
}
