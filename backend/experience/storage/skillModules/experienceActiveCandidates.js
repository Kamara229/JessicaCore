/** Select one active, highest-version facade per Skill before hydration.
 * Old versions never override new ones; inactive records never execute.
 * Corruption in the chosen version is rejected by the module hydrator.
 */
export function selectActiveExperienceCandidates(rows) {
  if(!Array.isArray(rows))return [];
  const selected=new Map();
  for(const row of rows) {
    if(row?.enabled!==true||row.status!=="published")continue;
    const id=typeof row.id==="string"?row.id.trim():"";
    const version=Number(row.version);
    if(!id||!Number.isSafeInteger(version)||version<1)
      throw new Error("Active Skill has invalid id or version");
    const previous=selected.get(id);
    if(!previous||version>Number(previous.version))
      selected.set(id,row);
    else if(version===Number(previous.version))
      throw new Error("Conflicting active Skill versions: "+id);
  }
  return Array.from(selected.values());
}
