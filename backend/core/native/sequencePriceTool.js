import {registerTool} from "../../tools/toolRegistry.js";
import {calculateSequentialPrice} from "./sequencePriceCalculation.js";
import {formatSequencePrice} from "./sequencePriceText.js";
export function evaluateSequencePrice(args={}){
 const r=calculateSequentialPrice(args);
 if(!r)return {success:false,text:"Недопустимые числовые условия"};
 return {success:true,data:{...args,...r},text:formatSequencePrice(r)};
}
registerTool({name:"sequential_price_percent",
 description:"Calculate sequential price percentage changes",
 arguments:{price:"Original price",increase:"Increase percentage",decrease:"Decrease percentage"},
 execute:evaluateSequencePrice});
