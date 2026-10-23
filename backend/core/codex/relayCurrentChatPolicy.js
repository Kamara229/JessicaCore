/**
 * A direct status message for the already selected ChatGPT chat is not a
 * request to create an independent Jessica test thread.
 */
export function directCurrentChatReply(message) {
    const text=String(message || "");
    const excludesNewTest=/не\s+(?:запускай\s+тест|создавай\s+новый\s+тестовый\s+чат)/iu.test(text);
    const sameChat=/в\s+(?:этой|той\s+же|текущей)\s+сессии|в\s+(?:этом\s+же|текущем|выбранном)\s+чате\s+chatgpt/iu.test(text);
    const asksToSend=/самостоятельно\s+отправь\s+(?:следующее\s+)?сообщение/iu.test(text);
    if (!excludesNewTest || !sameChat || !asksToSend) return null;
    const quote=[...text.matchAll(/«([^»]{10,350})»/gu)].map(m=>m[1].trim())
        .filter(s=>/^кодекс:\s+/iu.test(s));
    return quote.length===1 ? quote[0] : null;
}
