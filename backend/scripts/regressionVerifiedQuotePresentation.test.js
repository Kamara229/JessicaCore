import test from "node:test";
import assert from "node:assert/strict";
import {isVerifiedRequestedOriginalQuote} from "../core/language/verifiedQuotePreservation.js";
import {localizeResponse} from "../core/language/responseLanguageService.js";
import {enforcePresentationLanguage} from "../core/language/strictPresentationLanguage.js";
const task="Приведи 12 английских слов подряд из §3 Apache License 2.0 о прекращении патентных лицензий. Без перевода. Укажи официальный URL.";
const quote="licenses granted to You under this License for that Work shall terminate";
const good={success:true,validated:true,mode:"single",
 answerSource:"bounded-original-license-excerpt",
 text:"§3 Apache License 2.0: “"+quote+"”.\nОфициальный URL: https://www.apache.org/licenses/LICENSE-2.0.txt"};

test("verified English source quote survives Russian locale without translation",async()=>{
 assert.equal(isVerifiedRequestedOriginalQuote(good,task),true);
 let translations=0;
 const localized=await localizeResponse(good,task,"ru",async()=>{translations++;throw Error("no model budget");});
 assert.equal(translations,0);
 assert.equal(localized.localizationStatus,"verified-original-quote");
 assert.equal(localized.text,good.text);
 const final=enforcePresentationLanguage(localized,{task,language:"ru"});
 assert.equal(final.success,true);
 assert.equal(final.text,good.text);
});
test("unverified English prose is not exempt from the language guard",()=>{
 for(const bad of [
 {...good,validated:false},
 {...good,success:false},
 {...good,answerSource:"native-fallback"},
 {...good,text:"§3 Apache License 2.0: “some unrelated original English patent quote goes here and another example”.\nОфициальный URL: https://www.apache.org/licenses/LICENSE-2.0.txt"}
 ]) {
   assert.equal(isVerifiedRequestedOriginalQuote(bad,task),false);
 }
 const english={success:true,mode:"single",text:"Here is a complete explanation of the licensing conditions in plain English."};
 assert.equal(enforcePresentationLanguage(english,{task:"Объясни простыми словами",language:"ru"}).success,false);
});
