/**
 * A short, standalone "remember a number and word" instruction needs only
 * acknowledgement. The Android chat archive is the source of truth; the
 * backend must not substitute an answer from an earlier unrelated turn.
 *
 * This is not server-side persistent memory or evidence of a learned Skill.
 */
export function acknowledgeMemoryInstruction(task) {
    if (typeof task !== "string") return null;
    const current=task.trim();
    if (current.length>220) return null;
    const request=/^запомни\s+число\s+[-+]?\d{1,18}\s+и\s+(?:кодовое\s+)?слово\s+[\p{L}-]{2,40}\s*[.!?]?\s*(?:не\s+(?:повторяй|называй|указывай)\s+их\s+в\s+ответе\s*[.!?]?)?$/iu;
    if (!request.test(current)) return null;
    return {
        success:true,
        status:"COMPLETED",
        mode:"conversation_ack",
        text:"Запомнила.",
        learning:{
            success:false,queued:false,skipped:true,
            reason:"local_chat_context_not_verified_experience"
        }
    };
}
