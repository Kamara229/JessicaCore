/*
 * Source text can include unrelated help and application templates after
 * the explicitly marked end of an operative document. Exclude those
 * appended instructions from AI evidence context, but keep the original
 * fetched document untouched for URL-attributed quotation verification.
 */
export function selectOperativeSourceText(content) {
    const text=String(content||"");
    const normalized=text.toUpperCase();
    const heading=normalized.indexOf("TERMS AND CONDITIONS FOR USE");
    const marker="END OF TERMS AND CONDITIONS";
    const end=normalized.indexOf(marker);
    if(heading>=0 && end>heading+250) {
        return text.slice(0,end+marker.length);
    }
    return text;
}
