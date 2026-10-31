export function parseSafeAccountTransferTask(task) {
 const t=String(task||"");
 if(t.length<60||t.length>700)return null;
 if(!/PL\/pgSQL/iu.test(t))return null;
 if(!/(?:блокируй|заблокируй)\s+оба\s+сч[её]та/iu.test(t))return null;
 if(!/по\s+ID/iu.test(t)||!/сумм[уы]\s*>\s*0/iu.test(t))return null;
 if(!/наличи[ея]/iu.test(t)||!/остаток/iu.test(t))return null;
 if(!/без\s+COMMIT\s+внутри/iu.test(t))return null;
 return {};
}
