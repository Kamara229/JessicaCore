import {normalizeConversation} from "./conversationContext.js";

/**
 * Resolve narrow, explicit earlier-word questions from this chat's USER turns.
 * Never ask web search or accept a prior assistant guess as memory evidence.
 * The target command and older command must both be visible in the supplied
 * local chat excerpt, otherwise fail closed rather than inventing a word.
 */
export function recallWordBeforeCommand(task, conversation) {
    const question=String(task || "").trim();
    if (question.length>300) return null;
    const asked=/^какое\s+слово\s+я\s+просил(?:а)?\s+(написать|вывести|назвать)\s+(?:(строго|только|ровно)\s+)?одним\s+словом\s+перед\s+командой\s+(написать|вывести|ответить)\s+([\p{L}\p{M}]{1,40})\s*\?\s*(?:верни\s+ответ\s+дословно\s*\.)?$/iu.exec(question);
    if (!asked) return null;

    const verbMap={написать:"напиши",вывести:"выведи",назвать:"ответь",ответить:"ответь"};
    const olderVerb=verbMap[asked[1].toLowerCase()];
    const olderQualifier=asked[2]?.toLowerCase() || null;
    const markerVerb=verbMap[asked[3].toLowerCase()];
    const markerWord=asked[4].toLowerCase();

    const userCommands=normalizeConversation(conversation,{followup:true})
        .filter(item=>item.role==="user")
        .map(item=>{
            const parsed=/^(ответь|напиши|выведи)\s+(?:(строго|только|ровно)\s+)?(?:одним\s+словом|одно\s+слово)\s*:\s*([\p{L}\p{M}]{1,40})\s*[.!?]?$/iu.exec(item.content);
            return parsed?{verb:parsed[1].toLowerCase(),qualifier:parsed[2]?.toLowerCase() || null,word:parsed[3]}:null;
        });
    // Find the LATEST command whose actual requested output matches the
    // explicitly named marker, never an arbitrary older mention of that word.
    let marker=-1;
    for(let i=userCommands.length-1;i>=0;i--) {
        const command=userCommands[i];
        if(command && command.verb===markerVerb && command.word.toLowerCase()===markerWord) {
            marker=i;break;
        }
    }
    let previous=null;
    for(let i=marker-1;i>=0;i--) {
        const command=userCommands[i];
        if(command?.verb===olderVerb && (!olderQualifier || command.qualifier===olderQualifier)) {
            previous=command;break;
        }
    }
    const known=Boolean(previous);
    return {
        success:known,
        status:known?"COMPLETED":"NEEDS_CLARIFICATION",
        needsClarification:!known,
        mode:"conversation_recall",
        text:known?previous.word:"В доступной истории чата нет подтверждения запрошенной команды.",
        learning:{success:false,queued:false,skipped:true,reason:"conversation_only_not_reusable_experience"}
    };
}
