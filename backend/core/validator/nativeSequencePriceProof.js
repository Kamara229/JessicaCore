import {parseSequencePriceTask} from "../native/sequencePriceIntent.js";
import {evaluateSequencePrice} from "../native/sequencePriceTool.js";
export function verifiedNativeSequencePrice(task,plan,run,answer){
 const p=parseSequencePriceTask(task);
 if(!p || plan?.intent!=="native_sequential_price_percent" || answer?.source!=="tool")return false;
 const proof=evaluateSequencePrice(p);
 return proof.text===answer.text && proof.text===run?.results?.[0]?.text;
}
