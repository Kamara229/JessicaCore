import {getAIProviderAvailability} from "./providerAvailability.js";
import {getRetryAfterDetails} from "./retry/retryAfter.js";

// A second model has its own quota. Do not retry the same exhausted model,
// and never switch on validation, content, network or short rate errors.
export const QUOTA_FALLBACK_MODEL="openai/gpt-oss-120b";
const PRIMARY="openai/gpt-oss-20b";
let alternateBlockedUntil=0;

function dailyQuota(error){
  if(error?.code==="AI_PROVIDER_COOLDOWN")
    return getAIProviderAvailability().reason==="daily-quota";
  const message=String(error?.message||"").toLowerCase();
  return error?.status===429 &&
    /(tokens per day|requests per day|\(tpd\)|\(rpd\))/.test(message);
}

function blockAlternative(error){
  if(error?.status!==429)return;
  const retry=getRetryAfterDetails(error)?.milliseconds;
  const timeout=Number.isFinite(retry) ? retry : 60_000;
  alternateBlockedUntil=Math.max(alternateBlockedUntil,
    Date.now()+Math.max(60_000,Math.min(timeout,900_000)));
}

export async function withDailyQuotaModelFallback({
  primaryModel,primaryRequest,alternateRequest
}={}){
  if(typeof primaryRequest!=="function"||
    typeof alternateRequest!=="function")throw Error("Missing model request");
  try{
    return await primaryRequest();
  }catch(error){
    if(primaryModel!==PRIMARY||!dailyQuota(error))throw error;
    if(Date.now()<alternateBlockedUntil)throw error;
    console.warn("Jessica alternate model requested",{
      from:PRIMARY,to:QUOTA_FALLBACK_MODEL,reason:"daily-quota"
    });
    try{
      const result=await alternateRequest(QUOTA_FALLBACK_MODEL);
      console.log("Jessica alternate model succeeded",{
        model:QUOTA_FALLBACK_MODEL
      });
      return result;
    }catch(alternateError){
      blockAlternative(alternateError);
      console.warn("Jessica alternate model unavailable",{
        model:QUOTA_FALLBACK_MODEL,status:alternateError?.status||0
      });
      throw alternateError;
    }
  }
}
