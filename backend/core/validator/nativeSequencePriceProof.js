import {parseSequencePriceTask} from "../native/sequencePriceIntent.js";
import {evaluateSequencePrice} from "../native/sequencePriceTool.js";
export function verifiedNativeSequencePrice(task,plan,run,answer){
 const p=parseSequencePriceTask(task);
 if(!p || plan?.intent!=="native_sequential_price_percent" || answer?.source!=="tool" || answer?.tool!=="sequential_price_percent" || run?.success!==true)return false;
 const proof=evaluateSequencePrice(p);
 return proof.success===true && proof.text===answer.text && proof.text===run?.results?.[0]?.text &&
 JSON.stringify(proof.data)===JSON.stringify(run?.results?.[0]?.data) &&
 JSON.stringify(p)===JSON.stringify(plan.steps?.[0]?.arguments);
}
