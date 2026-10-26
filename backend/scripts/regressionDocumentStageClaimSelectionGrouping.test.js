import test from "node:test";
import assert from "node:assert/strict";
import {groupStageClaimSelections}
 from "../core/files/documentStageClaimSelectionGrouping.js";
import {renderMergedStageClaimSelections}
 from "../core/files/documentStageClaimSelectionMergeRender.js";

const claim=(text,window,evidence="«Основание "+window+"»")=>({
    stageNumber:1,windowNumber:window,sectionIndex:1,
    claim:"• Позиция автора: "+text+" [О"+window+":Ф1:Ф1]",
    evidence:"  Основание: "+evidence
});

test("only literal wording, not inferred semantic similarity, forms a group",()=>{
    const input=[
        claim("Срок составляет один год",1),
        claim("Срок составляет один год",2),
        claim("Срок равен одному году",3),
        claim("срок составляет один год",4)
    ];
    const grouped=groupStageClaimSelections(input);
    assert.equal(grouped.exactTextGroups,3);
    assert.equal(grouped.repeatedFormulations,1);
    assert.equal(grouped.groups[0].occurrences.length,2);
    assert.equal(grouped.semanticEquivalenceVerified,false);
    const rendered=renderMergedStageClaimSelections(grouped,4);
    assert.equal(rendered.split("Формулировка ").length,4);
    for(const item of input){
        assert.ok(rendered.includes(item.claim.match(/\[О\d+:Ф1:Ф1\]/u)[0]));
        assert.ok(rendered.includes(item.evidence));
    }
    assert.equal(rendered.split("Срок составляет один год").length,2);
});
test("malformed or citationless claims cannot be grouped",()=>{
    assert.equal(groupStageClaimSelections([]),null);
    assert.equal(groupStageClaimSelections([claim("Тезис",1),{
        ...claim("Тезис",2),claim:"• Позиция автора: Тезис"
    }]),null);
    assert.equal(groupStageClaimSelections([{
        ...claim("Тезис",1),evidence:""
    }]),null);
    assert.equal(groupStageClaimSelections([claim("Тезис",1)]).groups.length,1);
});
