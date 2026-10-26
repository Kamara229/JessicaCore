import {extractExactFactTokens} from "./documentExactFactTokens.js";

const QUOTES=/«([^»]{1,1200})»/gu;
const normalize=text=>text.replace(/\s+/gu," ").trim();
const OPERATORS={more:">",less:"<",notMore:"<=",notLess:">="};

function relationNearToken(text,value){
    const tokens=extractExactFactTokens(text);
    if(!tokens||tokens.limited)return null;
    const found=tokens.anchors.filter(token=>normalize(token.value)===normalize(value));
    if(found.length!==1)return null;
    const token=found[0];
    const prefix=text.slice(Math.max(0,token.startOffset-42),token.startOffset);
    const match=/(?:^|[^\p{L}])(?:(не)\s+)?(более|менее)\s*$/iu.exec(prefix);
    if(!match)return null;
    const type=match[2].toLowerCase()==="более"?"more":"less";
    return OPERATORS[match[1]?(type==="more"?"notMore":"notLess"):type];
}

// Deliberately narrow: contradicting explicit thresholds for the SAME cited number.
// A non-conflict is not proof of semantic equivalence or factual truth.
export function inspectQuotedNumericRelation(statement,evidence,sourceWindow,value){
    if([statement,evidence,sourceWindow,value].some(v=>typeof v!=="string"))
        return "unresolved";
    const claimed=relationNearToken(statement,value);
    if(!claimed)return "unresolved";
    const source=normalize(sourceWindow);
    const actual=new Set();
    for(const match of evidence.matchAll(QUOTES)){
        const quote=normalize(match[1]);
        if(!quote||!source.includes(quote))continue;
        const found=relationNearToken(quote,value);
        if(found)actual.add(found);
    }
    if(actual.size!==1)return "unresolved";
    const sourceOp=[...actual][0];
    if(claimed===sourceOp)return "aligned";
    if((claimed===">"&&(sourceOp==="<"||sourceOp==="<="))||
       (claimed==="<"&&(sourceOp===">"||sourceOp===">="))||
       (claimed===">="&&sourceOp==="<")||
       (claimed==="<="&&sourceOp===">"))return "contradiction";
    return "unresolved";
}
