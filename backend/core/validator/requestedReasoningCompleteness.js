// Guard missing explanations when the user explicitly requests justification.
// This detects omitted reasoning, NOT whether the reasoning itself is true.
const ASK=/(?:обосн(?:уй|уйте|ование)|аргументир(?:уй|уйте)|объясни(?:те)?\s+почему|поясни(?:те)?\s+почему|justify\s+(?:your\s+)?answer|explain\s+why)/iu;
const SUPPORT=/(?:потому\s+что|так\s+как|поскольку|следовательно|поэтому|значит|вытекает|противореч|предполож|допустим|согласно|отсюда|основан|обоснован|объяснен|доказ|посылк|контрпример|пример\s*:|если[^.!?\n]{2,200}то|\b(?:because|therefore|since|hence|suppose|assume)\b)/iu;
export function requestedReasoningMissing(task,answer){
  if(!ASK.test(String(task||"")))return false;
  const output=String(answer||"").trim();
  return output.length<20||!SUPPORT.test(output);
}
