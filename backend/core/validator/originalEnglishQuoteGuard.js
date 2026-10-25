/**
 * A request for verbatim English source wording is a presentation contract.
 * Russian paraphrases do not satisfy it, even when legal facts are correct.
 * Verify a real 8+ word original-source span; no inferred quotations.
 */
const englishRequest = task =>
    /(?:дословн|цитат|фраз|verbatim|quote)/iu.test(String(task||"")) &&
    /(?:английск|english)/iu.test(String(task||""));

const normalized = value => String(value||"").normalize("NFKC")
    .replace(/[^\p{L}\p{N}]+/gu," ").toLowerCase().trim()
    .replace(/\s+/gu," ");

function englishPhrases(value) {
    const text=String(value||"").replace(/https?:\/\/\S+/gu," ");
    const lines=text.split(/[\n\r]+/gu);
    const phrases=[];
    for(const line of lines) {
        const matches=line.match(/[A-Za-z]+(?:['-][A-Za-z]+)?(?:\s+[A-Za-z]+(?:['-][A-Za-z]+)?){7,}/gu)||[];
        for(const match of matches) {
            const words=normalized(match).split(" ");
            if(words.length>=8 && words.length<=20)phrases.push(words.join(" "));
            else for(let i=0;i+8<=words.length;i++)phrases.push(words.slice(i,i+8).join(" "));
        }
    }
    return phrases;
}

export function checkOriginalEnglishQuote(task,run,answer) {
    if(!englishRequest(task))return null;
    const content=(Array.isArray(run?.results)?run.results:[])
        .filter(item=>item?.tool==="web_fetch"&&item?.success===true)
        .map(item=>normalized(item?.data?.content)).filter(Boolean);
    const phrases=englishPhrases(answer);
    if(content.length && phrases.some(p=>content.some(source=>source.includes(p))))
        return null;
    return {valid:false,shouldRetry:true,
        reason:"Запрошена дословная английская цитата: ответ должен содержать подтверждённую оригиналом фразу, а не перевод"};
}
