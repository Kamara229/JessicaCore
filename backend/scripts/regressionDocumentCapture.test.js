import test from "node:test";
import assert from "node:assert/strict";
import { sanitizeSourceUrlInput } from "../tools/sourceUrlInput.js";
import { buildNativeDocumentFetchPlan } from "../core/planner/nativeDocumentFetchPlan.js";
import { getDocumentReceipt } from "../core/composer/documentReceipt.js";
import { shouldAnalyzeLearningTrace } from "../core/learning/trigger/triggerEligibility.js";
import { buildSubtaskDependencyContext } from "../core/subtask/subtaskDependencyContext.js";

const url="https://opensource.org/license/mit";
const task="Собрать полный текст MIT License, используя URL " + url;
const text="MIT License\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software.\n"
    + "The above copyright notice and this permission notice shall be included in all copies.\n".repeat(12);
const plan=buildNativeDocumentFetchPlan(task);
const run={success:true,results:[{tool:"web_fetch",success:true,data:{
    url,title:"The MIT License",content:text,truncated:false
}}]};

test("fetch-only native plan is created without planning AI",()=>{
    assert.equal(plan?.intent,"native_document_capture");
    assert.equal(plan?.steps?.[0]?.tool,"web_fetch");
    assert.equal(plan?.steps?.[0]?.arguments.url,url);
    assert.equal(buildNativeDocumentFetchPlan("Сравни MIT и Apache: "+url),null);
});

test("clean invisible characters from a copied document URL",()=>{
    const dirty=url+"\u2060\uFFFD";
    assert.equal(sanitizeSourceUrlInput(dirty),url);
    assert.equal(buildNativeDocumentFetchPlan(task+"\u2060\uFFFD")?.steps?.[0]?.arguments.url,url);
});

test("receipt uses only verified retrieved content without AI synthesis",()=>{
    const receipt=getDocumentReceipt(task,plan,run);
    assert.equal(receipt?.source,"document-receipt");
    assert.equal(receipt?.url,url);
    assert.ok(receipt?.text.includes("Permission is hereby granted"));
});

test("reject missing, truncated, misleading or wrong-source document",()=>{
    assert.equal(getDocumentReceipt(task,plan,{results:[]}),null);
    const badData=run.results[0].data;
    for(const change of [
        {truncated:true},{url:"https://opensource.org/license/apache-2.0"},
        {content:"MIT License"}, {content:"Apache License\n"+("Other content ".repeat(60))}
    ]){
        const altered={success:true,results:[{tool:"web_fetch",success:true,data:{...badData,...change}}]};
        assert.equal(getDocumentReceipt(task,plan,altered),null);
    }
});

test("verified document excerpts survive as dependency context",()=>{
    const receipt=getDocumentReceipt(task,plan,run);
    const input=buildSubtaskDependencyContext({dependencyResults:[{
        id:1,status:"COMPLETED",success:true,verified:true,
        answer:{text:receipt.text,source:receipt.source}
    }]});
    assert.ok(input.includes("Permission is hereby granted"));
    assert.ok(input.includes("The above copyright notice"));
});

const verified={status:"COMPLETED",success:true,verified:true,validation:{valid:true,degraded:false}};

test("partial complex tasks never launch positive Pattern Discovery",()=>{
    const trace={task:"Compare documents",result:{
        success:true,status:"PARTIAL",results:[
            verified,{status:"FAILED",success:false,verified:false}
        ]
    },statistics:{completed:1,failed:1}};
    assert.equal(shouldAnalyzeLearningTrace(trace),false);
});

test("all child outcomes must independently verify success",()=>{
    const trace={task:"Compare documents",result:{
        success:true,status:"COMPLETED",results:[verified,{...verified,verified:false}]
    },statistics:{completed:2}};
    assert.equal(shouldAnalyzeLearningTrace(trace),false);
    trace.result.results=[verified,{...verified}];
    assert.equal(shouldAnalyzeLearningTrace(trace),true);
});

test("AI fallback is not strong positive learning evidence",()=>{
    const single={result:{status:"COMPLETED",success:true,verified:true,
        validation:{valid:true,degraded:true}},statistics:{completed:1}};
    assert.equal(shouldAnalyzeLearningTrace(single),false);
    single.result.validation.degraded=false;
    assert.equal(shouldAnalyzeLearningTrace(single),true);
});

import "../tools/webFetchTool.js";
import { buildNativeSourceComparisonPlan } from "../core/planner/nativeSourceComparisonPlan.js";
import { tryCreateNativeSkillPlan } from "../core/planner/nativeSkillPlanner.js";

test("two independent source URLs bypass Groq Planner for a comparison", () => {
    const task="Сравни MIT License и Apache 2.0: https://opensource.org/license/mit и https://opensource.org/license/apache-2.0";
    const plan=buildNativeSourceComparisonPlan(task);
    assert.equal(plan?.intent,"native_two_source_comparison");
    assert.deepEqual(plan.steps.map(x=>x.arguments.url),[
        "https://opensource.org/license/mit",
        "https://opensource.org/license/apache-2.0"
    ]);
    const routed=tryCreateNativeSkillPlan({task,executionExperience:{skills:[]},planningContext:{}});
    assert.equal(routed.handled,true);
    assert.equal(routed.source,"native-two-source-comparison");
});

test("two source comparison normalizes invisible URL artifacts without inventing URLs",()=>{
    const task="Сравни документы https://opensource.org/license/mit\u2060\uFFFD и https://opensource.org/license/apache-2.0\u2060\uFFFD";
    const plan=buildNativeSourceComparisonPlan(task);
    assert.deepEqual(plan.steps.map(s=>s.arguments.url),[
        "https://opensource.org/license/mit",
        "https://opensource.org/license/apache-2.0"
    ]);
});

test("ambiguous comparisons do not force a two-source machine plan",()=>{
    assert.equal(buildNativeSourceComparisonPlan("Сравни лицензии MIT и Apache"),null);
    assert.equal(buildNativeSourceComparisonPlan("Сравни две страницы https://a.com"),null);
    assert.equal(buildNativeSourceComparisonPlan("Сравни https://a.com и https://a.com"),null);
    assert.equal(buildNativeSourceComparisonPlan("Прочитай https://a.com и https://b.com"),null);
});

import { tryNativeDecomposition } from "../core/decomposer/nativeDecomposer.js";
import { verifyPairedSourceContent } from "../core/validator/pairedSourceContent.js";
import { validateSourceContent } from "../core/validator/sourceContentValidator.js";

const mitBody = "MIT License. Permission is hereby granted, free of charge, to any person obtaining a copy. "
    + ("The above copyright notice and this permission notice shall be included in all copies. ").repeat(9);
const apacheBody = "Apache License, Version 2.0. 3. Grant of Patent License. "
    + ("You must give any other recipients of the Work or Derivative Works a copy of this License. ").repeat(9);
const comparisonInput = "Сравни MIT License и Apache License 2.0. "
    + "MIT: https://opensource.org/license/mit Apache: https://opensource.org/license/apache-2.0. "
    + "Сначала получи оба документа, затем сравни положения и приведи источники.";
const pairedPlan = buildNativeSourceComparisonPlan(comparisonInput);
const pairedRun = {success:true,results:[
    {tool:"web_fetch",success:true,data:{url:"https://opensource.org/license/mit",title:"MIT License",content:mitBody,truncated:false}},
    {tool:"web_fetch",success:true,data:{url:"https://opensource.org/license/apache-2.0",title:"Apache License",content:apacheBody,truncated:false}}
]};

test("two URL comparison is not split by then or then-after wording", () => {
    const decomposition=tryNativeDecomposition({task:comparisonInput});
    assert.equal(decomposition.handled,true);
    assert.equal(decomposition.decomposition.subtasks.length,1);
    assert.equal(decomposition.decomposition.subtasks[0].text,comparisonInput);
    assert.equal(decomposition.source,"native-source-pair");
});

test("pair acquisition checks both URLs and actual licence bodies", async () => {
    const checked=verifyPairedSourceContent(pairedPlan,pairedRun);
    assert.equal(checked.valid,true);
    const content=await validateSourceContent(comparisonInput,pairedPlan,pairedRun);
    assert.equal(content.valid,true);
    assert.ok(content.reason.includes("документа"));
});

test("pair acquisition fails when MIT body is empty or only site navigation", () => {
    for(const body of ["",("Menu Links Privacy About ".repeat(28))]){
        const run={...pairedRun,results:[
            {...pairedRun.results[0],data:{...pairedRun.results[0].data,content:body}},
            pairedRun.results[1]
        ]};
        assert.equal(verifyPairedSourceContent(pairedPlan,run).valid,false);
    }
});

test("pair acquisition rejects swapped or truncated documents", () => {
    const wrong={...pairedRun,results:[
        {...pairedRun.results[0],data:{...pairedRun.results[0].data,url:"https://opensource.org/license/apache-2.0"}},
        pairedRun.results[1]
    ]};
    assert.equal(verifyPairedSourceContent(pairedPlan,wrong).valid,false);
    const truncated={...pairedRun,results:[
        pairedRun.results[0],
        {...pairedRun.results[1],data:{...pairedRun.results[1].data,truncated:true}}
    ]};
    assert.equal(verifyPairedSourceContent(pairedPlan,truncated).valid,false);
});

import { buildComposerContext } from "../core/composer/composerContext.js";
import { buildPairedSourceContext } from "../core/composer/pairedSourceContext.js";
import { resolveComposerBudget } from "../ai/composerClient.js";

test("paired composer includes independently attributed excerpts from both sources",()=>{
    const context=buildComposerContext(comparisonInput,pairedPlan,pairedRun);
    assert.ok(context.includes("URL: https://opensource.org/license/mit"));
    assert.ok(context.includes("URL: https://opensource.org/license/apache-2.0"));
    assert.ok(context.includes("Permission"));
    assert.ok(context.includes("Grant of Patent License"));
    assert.ok(context.length<10000);
    assert.equal(buildPairedSourceContext(comparisonInput,{results:[pairedRun.results[0]]}),null);
});

test("long source pairs retain relevant separate excerpts without copying every page",()=>{
    const mit="MIT License Permission is hereby granted. "+("Other legal provisions ".repeat(190));
    const apache="Apache License Version 2.0. "+("General definitions ".repeat(200))
        +"Grant of Patent License with express patent terms. "+("Copyright notices ".repeat(150));
    const large={results:[
        {tool:"web_fetch",success:true,data:{url:"https://opensource.org/license/mit",content:mit}},
        {tool:"web_fetch",success:true,data:{url:"https://opensource.org/license/apache-2.0",content:apache}}
    ]};
    const trimmed=buildPairedSourceContext(comparisonInput+" Патентные условия, изменение файлов, уведомления.",large);
    assert.ok(trimmed.includes("https://opensource.org/license/mit"));
    assert.ok(trimmed.includes("https://opensource.org/license/apache-2.0"));
    assert.ok(trimmed.includes("Grant of Patent License"));
    assert.ok(trimmed.length<9000);
});

test("composer token budget applies only to paired sources",()=>{
    assert.deepEqual(resolveComposerBudget({pairedSources:true}),{
        max_completion_tokens:1400,reasoning_effort:"low"
    });
    assert.deepEqual(resolveComposerBudget({}),{});
});

import { scopeOfficialLicenseCandidates } from "../core/taskRunner/argumentResolver/officialLicenseScope.js";
import { requiresVerifiedComparison, usePairedComparisonContext } from "../core/composer/comparisonEvidencePolicy.js";

test("strict OSI source matching separates BSD licence variants",()=>{
    const sources=[
      {title:"BSD 2-Clause",url:"https://choosealicense.com/licenses/bsd-2-clause/"},
      {title:"The 3-Clause BSD License",url:"https://opensource.org/license/bsd-3-clause"},
      {title:"The 2-Clause BSD License",url:"https://opensource.org/license/bsd-2-clause"}
    ];
    const task="Используй официальные тексты Open Source Initiative";
    assert.deepEqual(scopeOfficialLicenseCandidates("BSD 2-Clause",task,sources).map(x=>x.url),
        ["https://opensource.org/license/bsd-2-clause"]);
    assert.deepEqual(scopeOfficialLicenseCandidates("BSD 3-Clause",task,sources).map(x=>x.url),
        ["https://opensource.org/license/bsd-3-clause"]);
    assert.deepEqual(scopeOfficialLicenseCandidates("BSD 2-Clause",task,[sources[0]]),[]);
});

test("source-based document comparisons cannot be deemed complete by a fallback",()=>{
    assert.equal(requiresVerifiedComparison("Сравни две лицензии",{evidence:{mode:"source_content"}}),true);
    assert.equal(requiresVerifiedComparison("Скачай документ",{evidence:{mode:"source_content"}}),false);
});

test("AI-planned two-source comparisons receive the same small evidence budget",()=>{
    const arbitraryPlan={intent:"compare_bsd_licenses",evidence:{mode:"source_content"}};
    assert.equal(usePairedComparisonContext("Сравни две лицензии",arbitraryPlan,pairedRun),true);
    const context=buildComposerContext("Сравни две лицензии",arbitraryPlan,pairedRun);
    assert.ok(context.includes("ИСТОЧНИК 1"));
    assert.ok(context.includes("ИСТОЧНИК 2"));
    assert.ok(context.length<10000);
    assert.equal(usePairedComparisonContext("Сравни две лицензии",arbitraryPlan,
        {results:[pairedRun.results[0]]}),false);
});

import { checkPairedClaimCoverage } from "../core/validator/claim/pairedClaimCoverage.js";
import { hasVerifiedPatternEvidence } from "../core/learning/patternExtraction/patternEvidencePolicy.js";

test("unverified second-source claims cannot validate comparative table",()=>{
    const claims=[
      {supported:true,sourceUrl:"https://opensource.org/license/mit"},
      {supported:true,sourceUrl:"https://opensource.org/license/mit"}
    ];
    const sample="| Criterion | MIT | Apache |\n|---|---|---|\n| Grant | Yes | Yes |\n| Patent | No | Yes |";
    assert.equal(checkPairedClaimCoverage({
        task:comparisonInput,plan:pairedPlan,
        taskRunResult:pairedRun,answer:sample,claims
    }).valid,false);
    const verified=[...claims.slice(0,1),
      {supported:true,sourceUrl:"https://opensource.org/license/apache-2.0"}];
    assert.equal(checkPairedClaimCoverage({
        task:comparisonInput,plan:pairedPlan,
        taskRunResult:pairedRun,answer:sample,claims:verified
    }).valid,true);
});

test("legacy comparison discovery requires two explicit source citations",()=>{
    const evidence={task:comparisonInput,answer:"The licences differ",
      result:{success:true,verified:true,status:"COMPLETED"},
      validation:{valid:true,degraded:false}};
    assert.equal(hasVerifiedPatternEvidence(evidence),false);
    evidence.answer="Compared sources: https://opensource.org/license/mit and https://opensource.org/license/apache-2.0";
    assert.equal(hasVerifiedPatternEvidence(evidence),true);
});

import { resolveComposerModel } from "../ai/composerModelPolicy.js";

test("grounded comparisons use a separate model budget",()=>{
    assert.equal(resolveComposerModel({pairedSources:false}),"openai/gpt-oss-20b");
    assert.ok(["openai/gpt-oss-20b","openai/gpt-oss-120b"].includes(resolveComposerModel({pairedSources:true})));
});

import { parseClaimEvidenceResponse } from "../core/validator/claim/claimEvidenceResponse.js";
import { matchesAttributedEvidence } from "../core/validator/claim/claimEvidenceMatcher.js";

test("claim quotation accepts Markdown links and emphasis only in attributed source",()=>{
    const first="SOURCE 1\nURL: https://creativecommons.org/licenses/by/4.0/deed.en\n"+
        "TITLE: CC BY 4.0\nSNIPPET 1:\n**Attribution** — You must give [appropriate credit](https://example.org/credit), "+
        "provide a link to the license, and [indicate if changes were made](https://example.org/changes).";
    const second="SOURCE 2\nURL: https://creativecommons.org/licenses/by-sa/4.0/deed.en\n"+
        "TITLE: CC BY-SA 4.0\nSNIPPET 1:\n**ShareAlike** — You must distribute your contributions under the same license.";
    const cite="Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.";
    const found={claim:"CC BY requires attribution",supported:true,evidence:cite,
        sourceUrl:"https://creativecommons.org/licenses/by/4.0/deed.en"};
    assert.equal(matchesAttributedEvidence(found,first+"\n\n"+second),true);
    const mistaken={...found,sourceUrl:"https://creativecommons.org/licenses/by-sa/4.0/deed.en"};
    assert.equal(matchesAttributedEvidence(mistaken,first+"\n\n"+second),false);
    const parsed=parseClaimEvidenceResponse(JSON.stringify({claims:[found,mistaken]}),first+"\n\n"+second);
    assert.deepEqual(parsed.claims.map(c=>c.supported),[true,false]);
});

test("fabricated or very short source quotes do not become verified claims",()=>{
    const text="SOURCE 1\nURL: https://example.com/a\nSNIPPET 1:\nThe material may be copied and distributed with credit.";
    assert.equal(matchesAttributedEvidence({
        supported:true,sourceUrl:"https://example.com/a",evidence:"Permission is granted to create patents"
    },text),false);
    assert.equal(matchesAttributedEvidence({
        supported:true,sourceUrl:"https://example.com/a",evidence:"copied"
    },text+"\nSOURCE 2\nURL: https://example.com/b\nSNIPPET 1: Other text"),false);
});

test("short exact version number remains valid for one source",()=>{
    const original="SOURCE 1\nURL: https://example.com/release\nSNIPPET 1:\nLatest version is v1.2.3";
    assert.equal(matchesAttributedEvidence({
        supported:true,sourceUrl:"https://example.com/release",evidence:"v1.2.3"
    },original),true);
    assert.equal(matchesAttributedEvidence({
        supported:true,sourceUrl:"https://example.com/wrong",evidence:"v1.2.3"
    },original),false);
});

import { buildOriginalSourceEvidenceContext } from "../core/validator/claim/claimEvidenceMatcher.js";

test("claim excerpt missing from compact context is verified against actual fetched source only",()=>{
    const actual=[
        {tool:"web_fetch",success:true,data:{url:"https://example.com/source-a",
            content:"**Attribution** — You must give [appropriate credit](https://example.com/help), provide a link to the license, and indicate if changes were made."}},
        {tool:"web_fetch",success:true,data:{url:"https://example.com/source-b",
            content:"**ShareAlike** — You must distribute your contributions under the same license as the original."}}
    ];
    const originals=buildOriginalSourceEvidenceContext({results:actual});
    const compact="SOURCE 1\nURL: https://example.com/source-a\nSNIPPET 1: License summary without specific attribution terms.\nSOURCE 2\nURL: https://example.com/source-b\nSNIPPET 1: Summary.";
    const claim={claim:"Attribution is required",supported:true,
        sourceUrl:"https://example.com/source-a",
        evidence:"Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made."};
    assert.equal(parseClaimEvidenceResponse(JSON.stringify({claims:[claim]}),compact).claims[0].supported,false);
    assert.equal(parseClaimEvidenceResponse(JSON.stringify({claims:[claim]}),compact,originals).claims[0].supported,true);
    const forged={...claim,sourceUrl:"https://example.com/source-b"};
    assert.equal(parseClaimEvidenceResponse(JSON.stringify({claims:[forged]}),compact,originals).claims[0].supported,false);
});
