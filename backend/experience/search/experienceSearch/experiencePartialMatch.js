import {canonicalizeExperienceTokens} from "./experienceText.js";

const NOISE=new Set("a an and the of in to on for with from by or one two three its be is are does do when how which what they this that both x y between into under using each where if not по с на при есть как это что двух первый две затем".split(" "));
const DOMAIN=new Set(["patent","license","clause","legal","terminate","rights","grant"]);

/** Partial matches are evidence only when two informative terms overlap,
 * including a legal-domain term. Generic one-word matches cannot trigger
 * a legal Skill. Exact phrase matches retain priority in the caller.
 */
export function partialExperiencePhraseMatch(taskTokens,phrases,phraseScore){
    let score=0,terms=[];
    const taskSet=new Set(taskTokens.filter(x=>!NOISE.has(x)));
    for(const phrase of phrases){
        const candidate=[...new Set(canonicalizeExperienceTokens(phrase).filter(x=>!NOISE.has(x)))];
        const overlap=candidate.filter(x=>taskSet.has(x));
        if(candidate.length<3||overlap.length<2||
            !overlap.some(x=>DOMAIN.has(x)))continue;
        const partialScore=phraseScore*Math.min(0.6,(overlap.length/candidate.length)*1.25);
        if(partialScore>score){score=partialScore;terms=overlap;}
    }
    return {score,terms};
}
