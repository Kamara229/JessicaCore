import test from "node:test";
import assert from "node:assert/strict";
import {repairComparisonCellValue} from "../core/composer/comparisonSemanticRepair.js";

const licenseA=[
 "2. Grant of Copyright License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, sublicense, and distribute the Work.",
 "4. Redistribution. You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work, excluding those notices that do not pertain to any part of the Derivative Works.",
 "If the Work includes a NOTICE text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file, excluding those notices that do not pertain to any part of the Derivative Works."
].join(" ");
const licenseB="Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the copyright notice, conditions and disclaimer in the documentation and/or other materials provided with the distribution.";

test("use: distinguish irrevocable copyright grant from patent litigation",()=>{
 const claim="Разрешено использование и изменение; предоставляется отзываемая (except in case of litigation) авторская лицензия.";
 const out=repairComparisonCellValue("use",claim,licenseA);
 assert.match(out,/безотзывную авторскую лицензию/iu);
 assert.doesNotMatch(out,/except in case of litigation/iu);
});

test("notices: do not infer whole-file NOTICE preservation",()=>{
 const claim="Требуется сохранять исходный файл **NOTICE** и включать его текст при распространении.";
 const out=repairComparisonCellValue("notices",claim,licenseA);
 assert.match(out,/читаемую копию/iu);
 assert.match(out,/относящихся к ней атрибуционных уведомлений/iu);
 assert.doesNotMatch(out,/сохранять исходный файл/iu);
});

test("notices: discard a separable unproven absence while retaining duties",()=>{
 const claim="Нужно сохранять копирайт-уведомления, список условий и отказ от ответственности; отдельный файл NOTICE не требуется. [Источник]";
 const out=repairComparisonCellValue("notices",claim,licenseB);
 assert.match(out,/Нужно сохранять копирайт-уведомления/iu);
 assert.doesNotMatch(out,/отдельный файл NOTICE не требуется/iu);
});

test("notices: leave evidence-qualified statements unchanged",()=>{
 const claim="Требуется сохранять копирайт-уведомления и отказ от ответственности.";
 assert.equal(repairComparisonCellValue("notices",claim,licenseB),claim);
});
