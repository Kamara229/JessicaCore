/** Non-array or malformed history must never masquerade as a new Skill. */
export function requireExperienceHistoryRows(rows){
    if(!Array.isArray(rows) || rows.some(row =>
        !row || typeof row!=="object" || Array.isArray(row)))
        throw new Error("Experience history result is incomplete or malformed");
    return rows;
}
