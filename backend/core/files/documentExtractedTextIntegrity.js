/**
 * Preserve the exact characters delivered by the file-format extractor.
 * The only historical cleanup is removal of embedded U+0000 characters.
 * This does NOT make a PDF or DOCX extraction byte-for-byte identical to
 * its source file: the parser has already converted the native file format.
 */
export function prepareExtractedDocumentText(rawText) {
    if (typeof rawText!=="string")
        return {text:"",hasReadableContent:false,nullCharactersRemoved:0};
    let nullCharactersRemoved=0;
    for(const character of rawText)
        if(character==="\u0000")nullCharactersRemoved++;
    const text=nullCharactersRemoved ? rawText.replace(/\u0000/gu,"") : rawText;
    // Never use trim() on the returned text; every source offset must survive.
    return {text,hasReadableContent:text.trim().length>0,
        nullCharactersRemoved};
}
