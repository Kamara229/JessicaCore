import test from "node:test";
import assert from "node:assert/strict";
import {repairComparisonCellValue} from "../core/composer/comparisonSemanticRepair.js";
import {narrowComparisonRightsClaim} from "../core/composer/comparisonClauseScopedRows.js";
const source=[
"Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, sublicense, and distribute the Work.",
"You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in Source or Object form, provided that You meet the following conditions:",
"(a) You must give any recipients a copy of this License; (b) You must cause modified files to carry prominent notices; (c) You must retain all original notices."
].join(" ");
test("use row drops separate notice duties only with explicit copyright grant",()=>{
 const claim="Разрешено использовать и создавать производные. Требуется сохранять уведомления о внесённых изменениях.";
 const value=repairComparisonCellValue("use",claim,source);
 assert.match(value,/безотзывная авторская лицензия/iu);
 assert.doesNotMatch(value,/требуется сохранять/iu);
 assert.equal(narrowComparisonRightsClaim("use",claim,"No copyright grant."),claim);
});
test("distribution row stops treating multiple duties as one cited fact",()=>{
 const claim="Разрешается распространять оригинал и производные при условии: (a) предоставить копию лицензии; (b) пометить изменённые файлы; (c) сохранить уведомления.";
 const value=repairComparisonCellValue("distribution",claim,source);
 assert.match(value,/исходной или объектной форме/iu);
 assert.doesNotMatch(value,/пометить/iu);
 assert.equal(narrowComparisonRightsClaim("distribution",claim,"No redistribution grant."),claim);
});
