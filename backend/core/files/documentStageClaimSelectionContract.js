/** Validate a strictly extractive selection of source-authenticated claims. */
export function parseStageClaimSelection(raw,page){
    if(!page||!Array.isArray(page.claims)||!page.claims.length||
       typeof raw!=="string"||raw.length>4000)return null;
    let decoded;
    try{decoded=JSON.parse(raw);}catch{return null;}
    if(!decoded||typeof decoded!=="object"||Array.isArray(decoded)||
       Object.keys(decoded).length!==1||
       !Array.isArray(decoded.selectedPositions))return null;
    const chosen=decoded.selectedPositions;
    if(!chosen.length||chosen.length>6)return null;
    let prior=0;
    for(const n of chosen){
        if(!Number.isSafeInteger(n)||n<=prior||n<1||
           n>page.claims.length)return null;
        prior=n;
    }
    return {selectedPositions:chosen,
        selected:chosen.map(n=>page.claims[n-1]),
        totalPageClaims:page.claims.length};
}
