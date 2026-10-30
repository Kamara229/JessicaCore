import {verifiedNativeIntegerProductAdd} from "../validator/nativeIntegerProductAddProof.js";

export function composeVerifiedIntegerProductAdd(task,plan,runResult){
  const result=runResult?.results?.[0];
  const answer={success:true,source:"native-integer-multiply-add-proof",text:result?.text};
  return verifiedNativeIntegerProductAdd(task,plan,runResult,answer)?answer:null;
}
