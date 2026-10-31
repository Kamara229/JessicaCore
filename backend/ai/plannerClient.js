import {getGroqClient} from "./groqClient.js";
import {executeAIWithRetry} from "./aiRetry.js";
import {withDailyQuotaModelFallback} from "./dailyQuotaModelFallback.js";

const MODEL="openai/gpt-oss-20b";
const MAX_COMPLETION=800;

/** Pure AI transport: JSON Plan. Semantic plan validation stays in Planner. */
export async function plannerChat(messages){
  if(!Array.isArray(messages)||messages.length===0||
    !messages.every(m=>m&&typeof m.role==="string"&&m.role.trim()&&
      typeof m.content==="string"&&m.content.trim()))
    throw Error("Planner messages некорректны");
  const options={
    model:MODEL,temperature:0,reasoning_effort:"low",include_reasoning:false,
    max_completion_tokens:MAX_COMPLETION,
    response_format:{type:"json_object"},tool_choice:"none",messages
  };
  const client=getGroqClient();
  return withDailyQuotaModelFallback({
    primaryModel:MODEL,
    primaryRequest:()=>executeAIWithRetry(
      ()=>client.chat.completions.create(options),
      {label:"Jessica Planner AI",maxAttempts:3}
    ),
    alternateRequest:(model)=>client.chat.completions.create(
      {...options,model},{maxRetries:0})
  });
}

export function getPlannerModel(){return MODEL;}
export function getPlannerMaxCompletionTokens(){return MAX_COMPLETION;}
