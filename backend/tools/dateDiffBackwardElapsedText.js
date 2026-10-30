/** Explain signed backward calendar arithmetic without treating it as a failed tool call. */
export function explainBackwardElapsed(startDate,endDate,signedDays){
  if(typeof startDate!=="string"||typeof endDate!=="string"||
     !Number.isSafeInteger(signedDays)||signedDays>=0) return null;
  return "Конечная дата "+endDate+" раньше начальной даты "+startDate+
    ". Расчёт: "+endDate+" − "+startDate+" = "+signedDays+
    " календарных дней (со знаком). В заданном направлении положительное "+
    "число прошедших дней определить нельзя; переставьте даты местами, "+
    "если нужен прямой интервал.";
}
