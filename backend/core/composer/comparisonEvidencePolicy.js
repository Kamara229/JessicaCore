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
