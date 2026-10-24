/**
 * Strictly literal, standalone one-word echo requests do not require a model.
 * Source of truth is the CURRENT user task, never a synonym, category, or
 * previous response. Avoid generating Experience from this echo contract.
 */
export function exactSingleWordReply(task) {
    if (typeof task !== "string") return null;
    const input=task.trim();
    if (!input || input.length > 180) return null;
    const match=input.match(/^(?:ответь|напиши|выведи)\s+(?:(?:только|строго|ровно)\s+)?(?:одним\s+словом|одно\s+слово)\s*:\s*([\p{L}\p{M}]{1,40})\s*[.!?]?$/iu);
    if (!match) return null;
    return {
        success:true,
        status:"COMPLETED",
        mode:"literal_word",
        text:match[1],
        learning:{
            success:false,queued:false,skipped:true,
            reason:"literal_instruction_no_generalizable_experience"
        }
    };
}
