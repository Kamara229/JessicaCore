/*
 * Separate model quota for source-grounded two-document comparisons.
 * Groq documents daily token limits per model. Keep the primary 20B
 * quota for normal tasks and let grounded comparisons use the 120B
 * reasoning model. This selects a model, not a claim of availability.
 */
const STANDARD_MODEL="openai/gpt-oss-20b";
const COMPARISON_MODEL="openai/gpt-oss-120b";
const ALLOWED=new Set([STANDARD_MODEL,COMPARISON_MODEL]);

export function resolveComposerModel({pairedSources=false}={}) {
    if(!pairedSources) return STANDARD_MODEL;
    const configured=String(process.env.GROQ_COMPARISON_MODEL || "").trim();
    return ALLOWED.has(configured)?configured:COMPARISON_MODEL;
}
