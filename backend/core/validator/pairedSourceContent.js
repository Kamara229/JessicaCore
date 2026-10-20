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
        // Do not mistake official how-to pages for the actual license.
        // Recognize Apache's canonical /licenses/LICENSE-2.0.{html,txt}.
        if (/\/licenses\/license-2\.0\.(?:html|txt)$/i.test(path) &&
            (!/apache license/i.test(content)
             || !/grant of patent license/i.test(content)
             || !/(?:redistribution|reproduction and distribution)/i.test(content))) {
            return {success:true,valid:false,shouldRetry:true,
                reason:"Текст Apache License 2.0 не обнаружен или неполон"};
        }
        // A legal comparison is not source-complete when HTML extraction
        // silently drops numbered redistribution clauses. Check the actual
        // source text, not merely its heading or broad license identity.
        if (/\/licenses\/license-2\.0\.(?:html|txt)$/i.test(path)) {
            const flat=content.replace(/\s+/g," ");
            const copyrightNotices=/\byou must retain\b[^.]{0,360}\battribution notices\b/i.test(flat);
            const conditionalNotice=/\bif the work includes a\b[^.]{0,110}\bnotice\b[^.]{0,350}\bmust include a readable copy\b[^.]{0,100}\battribution notices\b/i.test(flat);
            if(!copyrightNotices||!conditionalNotice) {
                return {success:true,valid:false,shouldRetry:true,
                    reason:"В загруженном тексте лицензии отсутствуют обязательные положения о сохранении уведомлений; источник неполон"};
            }
        }
        if (/\/(?:license|licenses)\/bsd-3-clause$/i.test(path) &&
            (!/redistribution and use in source and binary forms/i.test(content)
             || !/neither the name/i.test(content))) {
            return {success:true,valid:false,shouldRetry:true,
                reason:"Текст BSD 3-Clause не обнаружен или неполон"};
        }
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
