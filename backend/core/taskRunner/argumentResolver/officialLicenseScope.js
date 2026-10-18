/*
 * Preserve a user's named primary-source requirement during search.
 * Search snippets can rank mirrors above the requested publisher;
 * never silently replace OSI with a third-party licence explainer.
 */
export function scopeOfficialLicenseCandidates(query, task, candidates) {
    const all=Array.isArray(candidates)?candidates:[];
    const prompt=String(task || "");
    if(!/(?:open source initiative|opensource\.org)/i.test(prompt))return null;

    const fromOsi=all.filter(item=>{
        try {
            const host=new URL(String(item?.url||"")).hostname.toLowerCase();
            return host==="opensource.org" || host==="www.opensource.org";
        } catch {return false;}
    });

    const text=String(query||"").toLowerCase();
    // Disambiguate the two BSD variants even if "License" is absent.
    const bsd=text.match(/\bbsd[\s-]*(2|3)[\s-]*clause\b/i);
    if(!bsd)return fromOsi;

    const family=new RegExp("bsd[\\s-]*"+bsd[1]+"[\\s-]*clause","i");
    return fromOsi.filter(item=>{
        let path="";
        try {path=new URL(String(item.url)).pathname;} catch {}
        return family.test(String(item.title||"")+" "+path);
    });
}
