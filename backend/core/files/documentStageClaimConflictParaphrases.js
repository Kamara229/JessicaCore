import {LEGAL_CLAIM_FORMS,canonicalLegalClaimForm}
 from "./documentStageClaimConflictLexicon.js";
import {allowLegalClaimPolarityScope}
 from "./documentStageClaimConflictScopeGuard.js";

const PREFIX=/^• (?:В документе указано|Позиция автора|Аналитический вывод|Требует независимой проверки): /u;

/** Closed legal paraphrases: wording candidates, never verified semantic conflict. */
export function readStageClaimParaphrasePolarity(statement){
    if(typeof statement!=="string"||statement.length>2500||
       !PREFIX.test(statement))return null;
    const body=statement.replace(PREFIX,"").replace(/\s+/gu," ").trim();
    for(const {id,pattern} of LEGAL_CLAIM_FORMS){
        const match=pattern.exec(body);
        if(!match)continue;
        const before=body.slice(0,match.index);
        const after=body.slice(match.index+match[0].length);
        const negative=Boolean(match[1]);
        if(!allowLegalClaimPolarityScope({before,after,rule:id,negative}))
            continue;
        const normalized=canonicalLegalClaimForm(
            id,match[2].toLocaleLowerCase("ru"),negative);
        if(!normalized)continue;
        const key=(before+normalized+after).toLocaleLowerCase("ru").trim();
        if(!key)return null;
        return {key:"controlled:"+id+":"+key,negative,
            rule:"controlled-paraphrase:"+id};
    }
    return null;
}
