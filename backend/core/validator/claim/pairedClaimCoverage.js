/*
 * Comparative answer completeness gate.
 * A verifier checking only one document (or one of several table rows)
 * cannot certify the whole comparison.
 */
function normalizedUrl(value) {
    try {
        const url = new URL(String(value || "").trim());
        return url.hostname.toLowerCase() + url.pathname.replace(/\/+$/, "").toLowerCase();
    } catch { return ""; }
}

function answerRows(answer) {
    return String(answer || "").split("\n").filter(row => {
        const trimmed=row.trim();
        if (!trimmed.startsWith("|") || !trimmed.endsWith("|")) return false;
        if (/^\|[-|:\s]+\|$/.test(trimmed)) return false;
        return true;
    }).slice(1);
}

export function checkPairedClaimCoverage({task,plan,taskRunResult,answer,claims}={}) {
    const isComparison=/(?:сравн\p{L}*|сопостав\p{L}*|различи\p{L}*|compare|comparison|differences?)/iu
        .test(String(task||""));
    const fetches=(taskRunResult?.results||[]).filter(r=>r?.tool==="web_fetch"&&r?.success===true);
    if(!isComparison || plan?.evidence?.mode!=="source_content" || fetches.length!==2) {
        return {valid:true,reason:"Не применяется к этому типу задачи"};
    }
    const requested=new Set(fetches.map(r=>normalizedUrl(r?.data?.url)));
    if(requested.size!==2 || requested.has(""))return {
        valid:false,reason:"Не удалось подтвердить два независимых URL"
    };
    const present=new Set((claims||[]).filter(c=>c?.supported===true)
        .map(c=>normalizedUrl(c?.sourceUrl)).filter(Boolean));
    const missing=[...requested].filter(u=>!present.has(u));
    if(missing.length) return {
        valid:false,reason:"Не подтверждены утверждения по каждому из двух исходных документов"
    };
    const rows=answerRows(answer);
    const minimum=Math.max(2,rows.length*2);
    if((claims||[]).length<minimum) return {
        valid:false,reason:"Не проверены все существенные строки сравнительной таблицы ("+
            (claims||[]).length+" из "+minimum+")"
    };
    return {valid:true,reason:"Все сравниваемые источники и строки представлены в проверке"};
}
