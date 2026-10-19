/*
 * Deterministic two-source acquisition checks.
 * Confirms original URLs, nonempty substantive content and identity.
 * DOES NOT validate interpretation or factual claims; those remain
 * the responsibility of semantic/claim validation.
 */
function canonical(value) {
    try {
        const url = new URL(String(value || "").trim());
        return url.hostname.toLowerCase()
            + url.pathname.replace(/\/+$/, "").toLowerCase();
    } catch {
        return "";
    }
}

export function verifyPairedSourceContent(plan, taskRunResult) {
    if (!["native_two_source_comparison","native_known_license_comparison"].includes(plan?.intent)) return null;
    const steps = (plan.steps || []).filter(s => s.tool === "web_fetch");
    const results = (taskRunResult?.results || []).filter(r => r.tool === "web_fetch");
    if (steps.length !== 2 || results.length !== 2) {
        return {success:true,valid:false,shouldRetry:true,
            reason:"Для сравнения требуются два независимо загруженных документа"};
    }

    for (let index=0; index<2; index++) {
        const expected = canonical(steps[index].arguments?.url);
        const result = results[index], data = result?.data || {};
        const actual = canonical(data.url);
        const content = String(data.content || "").trim();
        if (!expected || !actual || expected !== actual || result?.success !== true) {
            return {success:true,valid:false,shouldRetry:true,
                reason:"Не подтверждён адрес документа " + (index+1)};
        }
        if (data.truncated === true || content.length < 350) {
            return {success:true,valid:false,shouldRetry:true,
                reason:"Неполное содержимое документа " + (index+1)};
        }
        const path = expected.toLowerCase();
        if (path.endsWith("/license/mit") &&
            (!/permission is hereby granted/i.test(content)
             || !/copyright notice/i.test(content))) {
            return {success:true,valid:false,shouldRetry:true,
                reason:"Текст лицензии MIT не обнаружен в документе"};
        }
        if (path.endsWith("/license/apache-2.0") &&
            (!/apache license/i.test(content)
             || !/grant of patent license/i.test(content))) {
            return {success:true,valid:false,shouldRetry:true,
                reason:"Текст лицензии Apache 2.0 не обнаружен в документе"};
        }
    }

    return {success:true,valid:true,shouldRetry:false,
        reason:"Оба документа проверены по URL и исходному тексту; выводы требуют отдельной проверки"};
}
