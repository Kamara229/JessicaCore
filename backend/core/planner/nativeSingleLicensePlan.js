import {KNOWN_LICENSE_SOURCES} from "./knownLicenseSources.js";

/**
 * A single named license with a primary-text question must not depend
 * on web_search ranking/selector. The known URL is only a location;
 * web_fetch and semantic validation must confirm the answer.
 */
export function buildNativeSingleLicensePlan(task) {
    const value=String(task||"");
    if(/https?:\/\//i.test(value) ||
        /(?:сравн\p{L}*|сопостав\p{L}*|compar(?:e|ison))/iu.test(value))return null;
    if(!/(?:официальн|дословн|цитат|английск|оригинал|первоисточник|фраз|патент|раздел|пункт|§|official|verbatim|quote|patent|section)/iu.test(value))
        return null;
    const found=KNOWN_LICENSE_SOURCES.filter(([,pattern])=>pattern.test(value));
    if(found.length!==1)return null;
    const [licenseId,,url]=found[0];
    return {
        intent:"native_single_license_primary_text",
        requiresTools:true,
        reasoningSummary:"Названный документ имеет канонический адрес; требуется чтение первоисточника.",
        evidence:{mode:"source_content",reason:"Факты и буквальные цитаты должны быть проверены по тексту лицензии."},
        steps:[{id:"fetch_named_license",tool:"web_fetch",arguments:{url}}],
        metadata:{plannerVersion:"native-single-license-v1",licenseId,
            generatedAt:new Date().toISOString()}
    };
}
