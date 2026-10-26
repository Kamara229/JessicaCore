import {groupStageClaimSelections} from "./documentStageClaimSelectionGrouping.js";
import {findStageClaimConflictCandidates} from "./documentStageClaimConflictCandidates.js";
import {triageStageClaimConflictCandidates} from "./documentStageClaimConflictTriage.js";

const id=(a,n)=>a+"→"+n;

/** Score explicitly labelled synthetic cases, NOT truth of customer documents. */
export function evaluateStageClaimConflictFixtures(cases){
    if(!Array.isArray(cases)||!cases.length||cases.length>48)
        return null;
    const names=new Set(),errors=[];
    let tp=0,fp=0,fn=0,reviewTierErrors=0;
    for(const sample of cases){
        if(typeof sample?.id!=="string"||
           !sample.id||names.has(sample.id)||
           !Array.isArray(sample.claims)||sample.claims.length<2||
           sample.claims.length>24||
           !Array.isArray(sample.expectedPairs)||
           sample.expectedPairs.length>12)return null;
        names.add(sample.id);
        const grouped=groupStageClaimSelections(sample.claims);
        const found=grouped&&findStageClaimConflictCandidates(grouped);
        const reviewed=found&&triageStageClaimConflictCandidates(found);
        if(!reviewed||found.candidateLimitReached)return null;
        const predicted=new Map();
        for(const pair of reviewed.candidates){
            const key=id(pair.affirmative.claim.windowNumber,
                pair.negative.claim.windowNumber);
            if(predicted.has(key))return null;
            predicted.set(key,pair.reviewTier);
        }
        const expected=new Map();
        for(const pair of sample.expectedPairs){
            if(!Number.isSafeInteger(pair?.affirmativeWindow)||
               !Number.isSafeInteger(pair?.negativeWindow)||
               !["primary-review","context-qualified-review"].includes(
                   pair?.reviewTier))return null;
            const key=id(pair.affirmativeWindow,pair.negativeWindow);
            if(expected.has(key))return null;
            expected.set(key,pair.reviewTier);
        }
        for(const [key,tier] of predicted){
            if(!expected.has(key)){
                fp++;errors.push({caseId:sample.id,type:"fixture-false-positive",
                    windowPair:key});
            }else{
                tp++;
                if(expected.get(key)!==tier){
                    reviewTierErrors++;
                    errors.push({caseId:sample.id,
                        type:"review-tier-mismatch",windowPair:key});
                }
            }
        }
        for(const key of expected.keys()){
            if(!predicted.has(key)){
                fn++;errors.push({caseId:sample.id,type:"fixture-missed-pair",
                    windowPair:key});
            }
        }
    }
    return {labelOrigin:"explicitly-annotated-synthetic-fixtures",
        casesEvaluated:cases.length,fixtureTruePositives:tp,
        fixtureFalsePositives:fp,fixtureMissedPairs:fn,
        fixtureReviewTierMismatches:reviewTierErrors,
        fixturePrecision:tp+fp?tp/(tp+fp):null,
        fixtureRecall:tp+fn?tp/(tp+fn):null,
        errors,realWorldAccuracyMeasured:false,
        semanticConflictVerified:false};
}
