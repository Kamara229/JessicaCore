import {registerTool} from "./toolRegistry.js";
import {extractInvalidCalendarRange} from "../core/native/invalidCalendarRange.js";
registerTool({
 name:"invalid_calendar_date",
 description:"Validate the existence of two calendar date operands.",
 arguments:{task:"The original date question."},
 execute(args){
  const match=extractInvalidCalendarRange(args?.task);
  if(!match)return {success:false,needsClarification:true,text:"Уточните даты."};
  const dates=match.invalidDates;
  return {success:true,text:"Дата "+dates.join(", ")+" не существует в календаре. Расчёт количества дней невозможен. Уточните дату.",
    data:{calendarValid:false,invalidDates:dates,fullDays:null}};
 }
});
