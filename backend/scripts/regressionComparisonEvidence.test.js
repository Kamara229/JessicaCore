import test from "node:test";
import assert from "node:assert/strict";
import { isUnsupportedNegativeComparisonClaim } from "../core/validator/claim/negativeComparisonClaim.js";
import { buildPairedSourceContext } from "../core/composer/pairedSourceContext.js";
import { buildClaimEvidenceContext } from "../core/validator/claim/claimEvidenceContext.js";
import { parseClaimEvidenceResponse } from "../core/validator/claim/claimEvidenceResponse.js";

const apacheUrl="https://www.apache.org/licenses/LICENSE-2.0.html";
const bsdUrl="https://opensource.org/license/bsd-3-clause";
const patentGrant="Each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable patent license.";
const termination="If You institute patent litigation against any entity, any patent licenses granted to You under this License for that Work shall terminate.";
const trademark="This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor.";
const apache=[
  "Apache License Version 2.0. Introduction. "+"Terms and conditions. ".repeat(58),
  "Terms for redistribution. "+"Redistribution copyright notices. ".repeat(44),
  "Grant of Patent License. "+patentGrant+" "+"Patent rights. ".repeat(69),
  "Conditions for patent claims. "+termination+" "+"Other legal terms. ".repeat(65),
  "Trademarks. "+trademark+" "+"End of license. ".repeat(40)
].join("\n\n");
const bsd="The 3-Clause BSD License. Redistribution and use in source and binary forms, with or without modification, are permitted provided that copyright notices are retained. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products.";
const run={results:[
 {tool:"web_fetch",success:true,data:{url:apacheUrl,title:"Apache",content:apache}},
 {tool:"web_fetch",success:true,data:{url:bsdUrl,title:"BSD",content:bsd}}
]};
const task="Сравни две лицензии Apache 2.0 и BSD 3-Clause: использование, распространение, уведомления, патентные положения и ограничения.";

test("plural negative provision claim does not become established from unrelated quotation",()=>{
 const claim={claim:"В тексте BSD 3-Clause отсутствуют патентные положения",
  evidence:"Redistribution and use in source and binary forms, with or without modification, are permitted"};
 assert.equal(isUnsupportedNegativeComparisonClaim(claim,true),true);
 assert.equal(isUnsupportedNegativeComparisonClaim({...claim,claim:"BSD 3-Clause не содержит отдельного патентного гранта"},true),true);
 const source="SOURCE 1\nURL: "+apacheUrl+"\nSNIPPET: "+patentGrant+
    "\n\nSOURCE 2\nURL: "+bsdUrl+"\nSNIPPET: "+bsd;
 const parsed=parseClaimEvidenceResponse(JSON.stringify({claims:[{
    ...claim,sourceUrl:"SOURCE 2",supported:true,reason:""
 }]}),source);
 assert.equal(parsed.success,true);
 assert.equal(parsed.claims[0].supported,false);
 assert.match(parsed.claims[0].reason,/Отрицательный вывод/);
});

test("composer sees real patent grant and distinct patent-litigation termination clause",()=>{
 const context=buildPairedSourceContext(task,run);
 assert.ok(context);
 assert.match(context,/Each Contributor hereby grants to You/);
 assert.match(context,/If You institute patent litigation/);
 assert.match(context,/does not grant permission to use the trade names/);
 assert.match(context,/3-Clause BSD License/);
 assert.ok(context.length<=10800);
});

test("claim validator retains the patent termination source in its bounded context",()=>{
 const c=buildClaimEvidenceContext({
   task,
   answer:"Apache License 2.0: patent grants terminate under patent litigation. BSD 3-Clause: patent terms were not verified.",
   taskRunResult:run
 });
 assert.equal(c.sourceCount,2);
 assert.match(c.sourceText,/SOURCE 1/);
 assert.match(c.sourceText,/SOURCE 2/);
 assert.match(c.sourceText,/patent licenses granted to You under this License/);
 assert.ok(c.sourceText.length<=10800);
});
