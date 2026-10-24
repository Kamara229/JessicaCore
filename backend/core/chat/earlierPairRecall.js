import { normalizeConversation } from "./conversationContext.js";
import {recallWordBeforeCommand} from "./earlierLiteralWordRecall.js";

/** Recall only user-provided pairs from this same chat; never guess from model replies. */
export function recallEarlierPair(task, conversation) {
    const wordRecall=recallWordBeforeCommand(task,conversation);
    if(wordRecall) return wordRecall;
    const question = String(task || "").trim();
    if (!/^какое число и слово я просил(?:а)? запомнить перед последней парой\?\s*(?:верни ответ дословно\.)?$/iu.test(question)) return null;
    const pairs = normalizeConversation(conversation, {followup:true})
        .filter(item => item.role === "user")
        .map(item => item.content.match(/^запомни число\s+(\d{1,18})\s+и\s+(?:кодовое\s+)?слово\s+([\p{L}-]{2,40})/iu))
        .filter(Boolean);
    const earlier = pairs.at(-2);
    const known = Boolean(earlier && pairs.at(-1));
    return {
        success:known,
        status:known ? "COMPLETED" : "NEEDS_CLARIFICATION",
        needsClarification:!known,
        mode:"conversation_recall",
        text:known ? earlier[1] + " и " + earlier[2] :
            "В доступной истории чата недостаточно двух подтверждённых пар.",
        learning:{success:false,queued:false,skipped:true,
            reason:"context_dependent_chat_recall"}
    };
}
