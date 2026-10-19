/* Structural row identifier; it never verifies a legal claim. */
const GROUPS={
use:/использ|изменен|модифиц|производн|\buse\b|modif/iu,
distribution:/распростран|дистрибуц|distribut/iu,
notices:/уведомлен|авторств|авторск|копирайт|copyright|notice|attribut/iu,
patents:/патент|patent/iu,
restrictions:/огранич|товарн|торгов|бренд|endors|trademark|restrict/iu,
warranty:/гарант|ответствен|warrant|liabilit/iu,
obligations:/обязанност|обязательств|требован|obligation|requirement/iu
};
export function matchComparisonCriterion(title,criteria){
 const value=String(title||"").replace(/\*\*/g,"").trim();
 const found=(criteria||[]).filter(c=>GROUPS[c.id]?.test(value));
 return found.length===1?found[0].id:null;
}
