const PREFIX=/^• (?:В документе указано|Позиция автора|Аналитический вывод|Требует независимой проверки): /u;
const OTHER_NEGATION=/(?<![\p{L}\p{N}_])не(?![\p{L}\p{N}_])/iu;

// Restricted patterns: only explicit negation of the SAME subject and predicate.
// Do not equate "can act" with "acts", or different cases, dates or conditions.
const FORMS=[
    {id:"entry-into-force",pattern:/(не\s+)?(вступил\s+в\s+силу)/iu},
    {id:"permission-verb",pattern:/(не\s+)?(разрешается)/iu},
    {id:"modal-capacity",pattern:/(не\s+)?(может\s+действовать)/iu},
    {id:"mandatory-adjective",pattern:/(не\s*)?(обязателен|обязательна|обязательно)/iu},
    {id:"right-to-act",pattern:/(не\s+)?(вправе|имеет\s+право|имеет\s+права)/iu}
];
function canonical(rule,form,negative){
    if(rule==="right-to-act"){
        // The "имеет права" form is only licensed with explicit negation.
        if(form==="имеет права"&&!negative)return null;
        return "вправе";
    }
    if(rule==="mandatory-adjective")return form;
    return form;
}

/** Whitelisted paraphrase/negation normalization, never semantic entailment. */
export function readStageClaimParaphrasePolarity(statement){
    if(typeof statement!=="string"||statement.length>2500||
       !PREFIX.test(statement))return null;
    const body=statement.replace(PREFIX,"").replace(/\s+/gu," ").trim();
    for(const {id,pattern} of FORMS){
        const m=pattern.exec(body);
        if(!m)continue;
        const before=body.slice(0,m.index),after=body.slice(m.index+m[0].length);
        if((before&&/[\p{L}\p{N}_]$/u.test(before))||
           (after&&/^[\p{L}\p{N}_]/u.test(after))||
           OTHER_NEGATION.test(before+" "+after))continue;
        const negative=Boolean(m[1]);
        const normalized=canonical(id,m[2].toLocaleLowerCase("ru"),negative);
        if(!normalized)continue;
        const key=(before+normalized+after).toLocaleLowerCase("ru").trim();
        if(!key)return null;
        return {key:"controlled:"+id+":"+key,negative,
            rule:"controlled-paraphrase:"+id};
    }
    return null;
}
