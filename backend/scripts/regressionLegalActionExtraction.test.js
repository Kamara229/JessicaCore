import test from "node:test";
import assert from "node:assert/strict";
import {extractLegalActionExcerpts as legal} from "../core/composer/sourceLegalActionExcerpts.js";
test("operative action, not headings, is extracted",()=>{
 const text="2. Copyright. Each Contributor hereby grants to You a perpetual, no-charge copyright license. 4. Redistribution. You may reproduce and distribute copies of the Work. 6. Trademarks. This License does not grant permission to use the trade names, trademarks, service marks, or product names.";
 assert.ok(legal(text,"use").some(q=>q.includes("hereby grants")));
 assert.ok(legal(text,"distribution").some(q=>q.includes("You may reproduce and distribute")));
 assert.ok(legal(text,"restrictions").some(q=>q.includes("does not grant permission")));
});
