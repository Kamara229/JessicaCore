import {createHash} from "node:crypto";

// JSONB changes object key order; hash canonical JSON, never insertion order.
function stable(value) {
    if(Array.isArray(value))return value.map(stable);
    if(value!==null&&typeof value==="object") {
        return Object.fromEntries(Object.entries(value)
            .filter(([,v])=>v!==undefined)
            .sort(([a],[b])=>a<b?-1:a>b?1:0)
            .map(([key,v])=>[key,stable(v)]));
    }
    return value;
}

export function hashExperienceModulePart(data) {
    if(!Array.isArray(data)||!data.length)
        throw new Error("Skill module part must be a nonempty array");
    const serialized=JSON.stringify(stable(data));
    if(typeof serialized!=="string")
        throw new Error("Skill module part cannot be serialized");
    return createHash("sha256").update(serialized,"utf8").digest("hex");
}
