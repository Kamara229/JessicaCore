import "../native/sequencePriceTool.js";
import {parseSequencePriceTask} from "../native/sequencePriceIntent.js";
export function buildNativeSequencePricePlan(task){
 const p=parseSequencePriceTask(task);if(!p)return null;
 return {intent:"native_sequential_price_percent",requiresTools:true,
 reasoningSummary:"Последовательные изменения процентов вычисляются без модели.",
 evidence:{mode:"none",reason:"Все величины приведены в задании."},
 experience:{used:false,source:"native-capability",skills:[]},
 steps:[{id:"native_sequential_price",tool:"sequential_price_percent",
   arguments:p,experienceUsed:false}],
 metadata:{plannerVersion:"native-sequential-price-v1"}};
}
