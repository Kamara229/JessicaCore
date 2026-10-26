import {compareStageClaimConflictContext} from "./documentStageClaimConflictContext.js";
import {readStageClaimParaphrasePolarity} from "./documentStageClaimConflictParaphrases.js";
const PREFIX=/^• (?:В документе указано|Позиция автора|Аналитический вывод|Требует независимой проверки): /u;
const VERB=/(?<![\p{L}\p{N}_])(не\s+)?(является|действует|допускается|предусмотрен|предусмотрена|предусмотрено|соответствует|обязан|обязана|обязаны)(?![\p{L}\p{N}_])/iu;
const EXTRA_NEGATION=/(?<![\p{L}\p{N}_])не(?![\p{L}\p{N}_])/iu;
const LIMIT=12;

function polarity(statement){
    if(typeof statement!=="string"||!PREFIX.test(statement))return null;
    const body=statement.replace(PREFIX,"").replace(/\s+/gu," ").trim();
    const match=VERB.exec(body);
    if(!match)return null;
    const start=match.index,end=start+match[0].length;
    const remainder=body.slice(0,start)+body.slice(end);
    if(EXTRA_NEGATION.test(remainder))return null;
    const key=(body.slice(0,start)+match[2]+body.slice(end))
        .toLocaleLowerCase("ru");
    return {key,negative:Boolean(match[1]),rule:"literal-negation"};
}

/** Conservative lexical candidates only; no inference that sources disagree in fact. */
export function findStageClaimConflictCandidates(grouping){
    if(!Array.isArray(grouping?.groups)||grouping.groups.length>768)
        return null;
    const index=new Map();
    for(const group of grouping.groups){
        const parsed=readStageClaimParaphrasePolarity(group?.statement)||
            polarity(group?.statement);
        if(!parsed)continue;
        if(!Array.isArray(group.occurrences)||!group.occurrences.length)
            return null;
        if(!index.has(parsed.key))index.set(parsed.key,{
            yes:[],no:[],rule:parsed.rule
        });
        const row=index.get(parsed.key);
        (parsed.negative?row.no:row.yes).push(...group.occurrences);
    }
    const candidates=[];
    let truncated=false;
    for(const record of index.values()){
        for(const affirmative of record.yes){
            for(const negative of record.no){
                if(affirmative.claim?.windowNumber===negative.claim?.windowNumber)
                    continue;
                if(candidates.length===LIMIT){truncated=true;break;}
                const pair={affirmative,negative,
                    reason:record.rule.startsWith("controlled-paraphrase:")?
                        "controlled-paraphrase-opposite-explicit-negation":
                        "same-wording-opposite-explicit-negation",
                    detectionRule:record.rule};
                const context=compareStageClaimConflictContext(pair);
                if(!context)return null;
                candidates.push({...pair,context});
            }
            if(truncated)break;
        }
        if(truncated)break;
    }
    return {candidates,possibleConflictPairs:candidates.length,
        controlledParaphrasePairs:candidates.filter(p=>
            p.reason==="controlled-paraphrase-opposite-explicit-negation").length,
        contextQualifiedPairs:candidates.filter(p=>
            p.context.differences.length>0).length,
        contextReviewPerformed:true,
        candidateLimitReached:truncated,
        scope:"selected-claims-explicit-negation-controlled-forms-different-windows",
        semanticConflictVerified:false,claimTruthVerified:false,
        unselectedClaimsChecked:false};
}
