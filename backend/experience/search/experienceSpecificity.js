import {canonicalizeExperienceTokens} from "./experienceSearch/experienceText.js";

/** Among near-equal matches, favor title concepts specific to this Skill.
 * Penalize a uniquely narrow title when the task never mentions its focus.
 * Never changes the matcher confidence or overrides an ambiguity by itself.
 */
export function titleSpecificityAdjustment(task,skill,skills){
    const taskTerms=new Set(canonicalizeExperienceTokens(task));
    const nameTerms=new Set(canonicalizeExperienceTokens(skill?.name||"").filter(x=>x.length>=4));
    const rare=[...nameTerms].filter(token=>
        skills.filter(item=>new Set(canonicalizeExperienceTokens(item?.name||"")).has(token)).length===1
    );
    if(rare.some(token=>taskTerms.has(token)))return 0.08;
    if(rare.some(token=>!taskTerms.has(token)))return -0.04;
    return 0;
}
