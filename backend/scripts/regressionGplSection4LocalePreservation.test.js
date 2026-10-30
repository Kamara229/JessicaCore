import test from "node:test";
import assert from "node:assert/strict";
import {isVerifiedRequestedOriginalQuote} from "../core/language/verifiedQuotePreservation.js";
import {localizeResponse} from "../core/language/responseLanguageService.js";
import {enforcePresentationLanguage} from "../core/language/strictPresentationLanguage.js";

const quote="conspicuously and appropriately publish on each copy an appropriate copyright notice";
const url="https://www.gnu.org/licenses/gpl-3.0.txt";
const text="§4 GPLv3: “"+quote+"”.\nОфициальный URL: "+url;
const tasks=[
 "Приведи дословную цитату из §4 GPL 3.0 о copyright notice. Укажи официальный URL",
 "Приведи дословную цитату из §4 GPLv3 о сохранении уведомлений об авторском праве. Укажи официальный URL. Не пересказывай"
];
const verified={success:true,validated:true,mode:"single",answerSource:"bounded-gpl-section4-excerpt",text};

test("exact GPLv3 section 4 quote is never translated or rewritten by Russian locale",async()=>{
 for(const task of tasks){
  assert.equal(isVerifiedRequestedOriginalQuote(verified,task),true);
  let translations=0;
  const result=await localizeResponse(verified,task,"ru",async()=>{
   translations++;return "§4 GPLv3: «неверный русский пересказ».\nОфициальный URL: "+url;
  });
  assert.equal(translations,0);
  assert.equal(result.localizationStatus,"verified-original-quote");
  assert.equal(result.text,text);
  const output=enforcePresentationLanguage(result,{task,language:"ru",override:false});
  assert.equal(output.success,true);
  assert.equal(output.text,text);
  assert.equal(output.text.includes(quote),true);
 }
});

test("no bypass for unverified, mistranscribed or unrelated quotations",()=>{
 const task=tasks[0];
 assert.equal(isVerifiedRequestedOriginalQuote({...verified,validated:false},task),false);
 assert.equal(isVerifiedRequestedOriginalQuote({...verified,answerSource:"composer"},task),false);
 assert.equal(isVerifiedRequestedOriginalQuote({...verified,text:text.replace("copyright notice","notice")},task),false);
 assert.equal(isVerifiedRequestedOriginalQuote(verified,"Переведи цитату на русский язык"),false);
 assert.equal(isVerifiedRequestedOriginalQuote({...verified,success:false},task),false);
});
