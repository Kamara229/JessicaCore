/*
 * Bounded, function-scoped Experience modules.
 * Parent payload stays a small searchable facade. The canonical full
 * Experience is validated by the existing Quality Inspector BEFORE save.
 */
export const MODULE_FIELDS=[
  "workflow","validationRules","constraints","strategy","sourcePriority",
  "successfulPatterns","failurePatterns","avoidPatterns","examples"
];
const MAX_BYTES=4096;
const MAX_MODULES=100;
const bytes=x=>Buffer.byteLength(JSON.stringify(x),"utf8");

export function packageExperienceModules(experience){
  if(!experience||typeof experience!=="object"||Array.isArray(experience))
    throw new Error("Skill facade input must be an object");
  const facade={...experience};
  const modules=[],counts={};
  for(const field of MODULE_FIELDS){
    const items=facade[field];
    if(!Array.isArray(items)||!items.length)continue;
    const parts=[];
    let chunk=[];
    for(const item of items){
      if(bytes([item])>MAX_BYTES)
        throw new Error("Oversize Skill module item in "+field);
      if(chunk.length&&bytes([...chunk,item])>MAX_BYTES){
        parts.push(chunk);chunk=[];
      }
      chunk.push(item);
    }
    if(chunk.length)parts.push(chunk);
    counts[field]=parts.length;
    parts.forEach((data,part)=>modules.push({moduleKey:field,part,data}));
    delete facade[field];
  }
  if(modules.length>MAX_MODULES)
    throw new Error("Skill module count exceeds "+MAX_MODULES);
  if(modules.length){
    if(Object.hasOwn(facade,"moduleManifest"))
      throw new Error("Refusing to overwrite existing module manifest");
    facade.moduleManifest={version:1,counts};
  }
  return {facade,modules};
}
