/*
 * A fragment from a retrieved page is not a verified comparison.
 * This pure policy prevents successful answers when AI synthesis or
 * claim validation is unavailable.
 */
export function requiresVerifiedComparison(task, plan) {
    const text=String(task||"");
    const comparison=/(?:сравн\p{L}*|сопостав\p{L}*|различи\p{L}*|\bcompar(?:e|ison)\b|\bdifferenc(?:e|es)\b)/iu.test(text);
    return comparison && (
        plan?.evidence?.mode==="source_content" ||
        plan?.intent==="native_two_source_comparison"
    );
}

export function usePairedComparisonContext(task,plan,run) {
    if (!requiresVerifiedComparison(task,plan)) return false;
    const fetches=Array.isArray(run?.results)
        ? run.results.filter(x=>x?.tool==="web_fetch"&&x?.success===true)
        : [];
    return fetches.length===2;
}
