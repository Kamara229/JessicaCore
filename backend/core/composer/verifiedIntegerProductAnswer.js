import {verifiedNativeIntegerProduct} from "../validator/nativeIntegerProductProof.js";

// An exact, executed product is already a complete answer. Do not call an AI
// composer when the tool output has passed the same proof used by Validator.
export function composeVerifiedIntegerProduct(task,plan,runResult){
  const result=runResult?.results?.[0];
  const answer={success:true,source:"native-integer-proof",text:result?.text};
  return verifiedNativeIntegerProduct(task,plan,runResult,answer)?answer:null;
}
