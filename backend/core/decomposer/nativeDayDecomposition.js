import {extractNativeDates} from "../native/dateUtils.js";
export function nativeDayDecomposition(task,memoryContext,memoryHints=[]) {
 const value=String(task||"").trim();
 if(value.length>500 || !value.startsWith("Сколько дней между ")) return null;
 if(extractNativeDates(value).length!==2 || /(?:затем|далее|после этого|и сколько|и вычисли|плюс|добавь|then|compare)/iu.test(value)) return null;
 return {handled:true,source:"native-day-range",decomposition:{
 originalTask:value,isComplex:false,memoryContext,subtasks:[{id:1,text:value,dependsOn:[],contributesToFinalAnswer:true,memoryHints}]}};
}