import test from "node:test";
import assert from "node:assert/strict";
import {searchExperience} from "../experience/search/experienceSearch.js";
import {canonicalizeExperienceTokens} from "../experience/search/experienceSearch/experienceText.js";

// Compact projections of three independently learned, published Supabase Skills.
const skills=[
  {
    "id": "legal-clause-comparison",
    "name": "Legal Clause Comparison",
    "category": "Legal Analysis",
    "description": "A reusable procedure for locating, extracting, and contrasting specific clauses across multiple legal documents.",
    "triggerPatterns": [
      "Compare clause X of Document A with clause Y of Document B",
      "Analyze differences in patent terms between two licenses",
      "Identify termination conditions for patent licenses in legal texts",
      "Does document A contain an explicit clause about Y?"
    ],
    "workflow": [
      "Identify the target documents and their official versions.",
      "Retrieve the full texts from authoritative sources.",
      "Extract the clauses and provide direct references."
    ],
    "enabled": true,
    "status": "published",
    "confidence": 0.82,
    "keywords": [
      "Compare clause X of Document A with clause Y of Document B",
      "Analyze differences in patent terms between two licenses",
      "Identify termination conditions for patent licenses in legal texts",
      "Does document A contain an explicit clause about Y?"
    ],
    "successfulPatterns": []
  },
  {
    "id": "license-clause-comparison",
    "name": "License Clause Comparison",
    "category": "Legal Analysis",
    "description": "Retrieve official license documents, locate specified clauses and compare their content.",
    "triggerPatterns": [
      "Compare clause X in license A with clause X in license B",
      "Analyze patent provisions in Apache 2.0 versus BSD 3-Clause",
      "Compare specific clauses of two legal documents",
      "Analyze termination or patent provisions in licenses"
    ],
    "workflow": [
      "Identify licenses to compare.",
      "Retrieve official license texts from authoritative domains.",
      "Compare their clauses and identify differences."
    ],
    "enabled": true,
    "status": "published",
    "confidence": 0.82,
    "keywords": [
      "Compare clause X in license A with clause X in license B",
      "Analyze patent provisions in Apache 2.0 versus BSD 3-Clause",
      "Compare specific clauses of two legal documents",
      "Analyze termination or patent provisions in licenses"
    ],
    "successfulPatterns": []
  },
  {
    "id": "license-patent-clause-comparison",
    "name": "License Patent Clause Comparison",
    "category": "Legal Document Analysis",
    "description": "Retrieve official license texts and compare patent clauses, scope and termination.",
    "triggerPatterns": [
      "Compare patent clauses of license A and license B",
      "Analyze patent provisions in open-source license X",
      "Identify termination conditions for patent rights in a license",
      "Analyze patent grant scope across licenses"
    ],
    "workflow": [
      "Identify official URLs of licenses.",
      "Fetch the full texts.",
      "Extract patent grants and termination clauses.",
      "Compare their conditions."
    ],
    "enabled": true,
    "status": "published",
    "confidence": 0.82,
    "keywords": [
      "Compare patent clauses of license A and license B",
      "Analyze patent provisions in open-source license X",
      "Identify termination conditions for patent rights in a license",
      "Analyze patent grant scope across licenses"
    ],
    "successfulPatterns": []
  }
];

test("specific patent License Skill wins three-way overlap in Russian",()=>{
    const query="Сравни патентные условия MIT License и Apache 2.0 по официальным текстам. Есть ли явная патентная лицензия и когда она прекращается? Укажи пункты и ссылки.";
    assert.equal(searchExperience(query,skills).experience?.id,"license-patent-clause-comparison");
    assert.deepEqual(canonicalizeExperienceTokens("патентные лицензии").slice(0,2),["patent","license"]);
});
test("legal clause comparison does not become an unrelated patent Skill",()=>{
    const query="Сравни юридические положения двух документов, укажи пункты и ссылки.";
    assert.equal(searchExperience(query,skills).experience?.id,"legal-clause-comparison");
});
test("general license comparison avoids unrelated patent-only specificity",()=>{
    const query="Compare license conditions for two open-source software packages.";
    assert.equal(searchExperience(query,skills).experience?.id,"license-clause-comparison");
});
test("unrelated tasks cannot acquire legal Skill from generic words",()=>{
    assert.equal(searchExperience("Как получить сертификат на продукт?",skills).found,false);
});
test("draft and disabled Skills never compete even when injected directly",()=>{
    const task="Compare patent clauses of Apache 2.0 and MIT License using official sources.";
    const gone=skills.map(skill=>({...skill,status:"draft"}));
    assert.equal(searchExperience(task,gone).found,false);
    const disabled=skills.map(skill=>({...skill,enabled:false}));
    assert.equal(searchExperience(task,disabled).found,false);
    assert.equal(searchExperience(task,skills).experience?.id,"license-patent-clause-comparison");
});
test("overlap without a unique title clue remains ambiguous",()=>{
    const same=[
      {id:"a",name:"Legal Clause Comparison",triggerPatterns:["Compare legal clauses"],keywords:["Compare legal clauses"],confidence:.82,enabled:true,status:"published"},
      {id:"b",name:"Legal Clause Comparison",triggerPatterns:["Compare legal clauses"],keywords:["Compare legal clauses"],confidence:.82,enabled:true,status:"published"}
    ];
    assert.equal(searchExperience("Compare legal clauses",same).found,false);
});
