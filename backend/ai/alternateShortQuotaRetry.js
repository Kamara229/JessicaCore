import {getRetryAfterDetails} from "./retry/retryAfter.js";
import {getAIRetryDelay} from "./retry/retryDelay.js";
import {isRetryableAIError} from "./retry/retryClassifier.js";
import {sleepAIWithAbort} from "./retry/retryAbort.js";

// A separate model may hit a short TPM/RPM limit while the primary
// daily quota is exhausted. Retry it once only when the provider gives
// a bounded near-term reset; do not loop on TPD, RPD, or request-shape 429.
export function isShortAlternateRateLimit(error) {
  if(error?.status!==429 || !isRetryableAIError(error))return false;
  const reason=String(error.message||"").toLowerCase();
  if(!/(tokens per minute|requests per minute|\(tpm\)|\(rpm\))/.test(reason))return false;
  const ms=getRetryAfterDetails(error)?.milliseconds;
  return Number.isFinite(ms) && ms>=0 && ms<=5000;
}

export async function requestAlternateWithShortRetry(
  request, model, {sleep=sleepAIWithAbort,signal}={}
) {
  if(typeof request!=="function")throw Error("Alternate request missing");
  try{
    return await request(model);
  }catch(error){
    if(!isShortAlternateRateLimit(error))throw error;
    const delay=getAIRetryDelay(error,1);
    await sleep(delay,signal);
    return request(model);
  }
}
