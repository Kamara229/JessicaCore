import test from "node:test";
import assert from "node:assert/strict";
import {isVerifiedRequestedOriginalQuote} from "../core/language/verifiedQuotePreservation.js";
import {localizeResponse} from "../core/language/responseLanguageService.js";
import {enforcePresentationLanguage} from "../core/language/strictPresentationLanguage.js";

const task="Приведи дословную английскую цитату из BSD 3-Clause о запрете использования имён авторов для продвижения. Укажи официальный URL";
const clause="Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.";
const url="https://opensource.org/license/bsd-3-clause";
const raw='"3. '+clause+'"\nSource: '+url;
const verified={success:true,validated:true,mode:"single",answerSource:"composer",text:raw};

test("verified BSD 3-Clause original English quote never passes through language translation",async()=>{
 assert.equal(isVerifiedRequestedOriginalQuote(verified,task),true);
 let calls=0;
 const localized=await localizeResponse(verified,task,"ru",async()=>{
  calls++;return '"3. Ни имя держателя авторских прав..."';
 });
 assert.equal(calls,0);
 assert.equal(localized.localizationStatus,"verified-original-quote");
 assert.equal(localized.text,raw);
 const outgoing=enforcePresentationLanguage(localized,{task,language:"ru"});
 assert.equal(outgoing.success,true);
 assert.equal(outgoing.text,raw);
});

test("BSD quote guard fails closed for non-verbatim, wrong URL, or unverified answers",()=>{
 const cases=[
  {...verified,validated:false},
  {...verified,success:false},
  {...verified,mode:"document"},
  {...verified,text:raw.replace(clause,"translated content")},
  {...verified,text:raw.replace(url,"https://wrong.example/bsd-3")},
  {...verified,text:raw+"\nA second \"quote\""}
 ];
 for(const c of cases)assert.equal(isVerifiedRequestedOriginalQuote(c,task),false);
 assert.equal(isVerifiedRequestedOriginalQuote(verified,"Переведи условие BSD 3-Clause на русский"),false);
 assert.equal(isVerifiedRequestedOriginalQuote(verified,"Приведи цитату из BSD 2-Clause"),false);
});

test("bare exact BSD3 clause survives Russian locale without a model call",async()=>{
 const text=clause+"  \nOfficial URL: "+url;
 const source={...verified,text};
 assert.equal(isVerifiedRequestedOriginalQuote(source,task),true);
 let calls=0;
 const localized=await localizeResponse(source,task,"ru",async()=>{calls++;return "Russian paraphrase";});
 assert.equal(calls,0);
 assert.equal(localized.text,text);
 assert.equal(localized.localizationStatus,"verified-original-quote");
 assert.equal(enforcePresentationLanguage(localized,{task,language:"ru"}).text,text);
 assert.equal(isVerifiedRequestedOriginalQuote({...source,validated:false},task),false);
 assert.equal(isVerifiedRequestedOriginalQuote({...source,text:text+"\nThird line"},task),false);
});

test("actual 20:49 official URL label cannot be translated",async()=>{
 const original=clause+"  \nURL: "+url;
 const validated={...verified,text:original};
 assert.equal(isVerifiedRequestedOriginalQuote(validated,task),true);
 let translated=0;
 const result=await localizeResponse(validated,task,"ru",async()=>{translated++;return "Недословный перевод";});
 assert.equal(translated,0);
 assert.equal(result.localizationStatus,"verified-original-quote");
 assert.equal(result.text,original);
 assert.equal(enforcePresentationLanguage(result,{task,language:"ru"}).text,original);
 assert.equal(isVerifiedRequestedOriginalQuote({...validated,text:original.replace("URL:","Unverified:")},task),false);
});
