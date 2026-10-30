import {fetchSpdxMitText} from "./canonicalMitSpdxText.js";
/** Fixed-host original MIT text extraction, independent of AI/TinyFish. */
const URL = "https://opensource.org/license/mit";
const END = "other dealings in the software";

function readableHtml(html) {
    const entities = {amp:"&",quot:String.fromCharCode(34),apos:String.fromCharCode(39),nbsp:" ",lt:"<",gt:">"};
    return String(html)
        .replace(/<(?:script|style|noscript)\b[^>]*>[\s\S]*?<\/(?:script|style|noscript)>/giu, " ")
        .replace(/<!--[\s\S]*?-->/gu, " ")
        .replace(/<[^>]+>/gu, " ")
        .replace(/&([a-z]+);/giu, (_, key) => entities[key.toLowerCase()] ?? " ")
        .replace(/&#(?:x([a-f0-9]+)|(\d+));/giu, (_, hex, decimal) => {
            const point = parseInt(hex ?? decimal, hex ? 16 : 10);
            return point >= 32 && point <= 0x10ffff ? String.fromCodePoint(point) : " ";
        })
        .replace(/\s+/gu, " ").trim();
}

export async function fetchCanonicalMitText(url, request = globalThis.fetch) {
    if (url !== URL) return null;
    const backup=()=>fetchSpdxMitText(request);
    try {
        const response = await request(URL, {
            method: "GET", redirect: "follow",
            headers: {Accept: "text/html", "Accept-Language": "en"},
            signal: AbortSignal.timeout(8500)
        });
        if (!response?.ok) return backup();
        const type = String(response.headers?.get?.("content-type") || "").toLowerCase();
        if (type && !type.includes("text/html")) return backup();
        const raw = await response.text();
        if (raw.length < 850 || raw.length > 900000) return backup();
        const text = readableHtml(raw);
        const lower = text.toLowerCase();
        const start = lower.indexOf("permission is hereby granted, free of charge");
        const end = lower.indexOf(END, start);
        if (start < 0 || end < start + 650) return backup();
        const operative = text.slice(start, end + END.length).trim();
        const check = operative.toLowerCase();
        const required = [
            "including without limitation the rights to use, copy, modify, merge, publish, distribute",
            "the above copyright notice and this permission notice shall be included",
            "the software is provided",
            "in no event shall the authors or copyright holders be liable"
        ];
        if (!required.every(item => check.includes(item)) || /\bpatent\b/u.test(check)) return backup();
        const content = "MIT License\n" + operative;
        return {success:true, text:"", data:{
            title:"The MIT License - Open Source Initiative",
            url:URL, content, truncated:false,
            originalLength:content.length, returnedLength:content.length
        }};
    } catch { return backup(); }
}
