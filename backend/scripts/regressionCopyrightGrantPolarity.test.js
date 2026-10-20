import test from "node:test";
import assert from "node:assert/strict";
import {correctCopyrightGrantPolarity} from "../core/composer/comparisonGrantPolarity.js";
import {contradictsExpressCopyrightGrant} from "../core/validator/claim/comparisonGrantPolarityGate.js";

const grant="Subject to the terms of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, sublicense, and distribute the Work.";
const wrong="Разрешено использовать и изменять при соблюдении условий; лицензия предоставляет бессрочную, всемирную, безвозмездную и отзываемую (except in case of litigation) авторскую лицензию.";

test("repair reverses a falsely revocable copyright grant",()=>{
 const actual=correctCopyrightGrantPolarity(wrong,grant);
 assert.match(actual,/безотзывную авторскую лицензию/iu);
 assert.doesNotMatch(actual,/except in case of litigation/iu);
 assert.equal(contradictsExpressCopyrightGrant({value:actual},[grant]),false);
});
test("validator independently rejects wrong copyright revocability even with verbatim quotes",()=>{
 assert.equal(contradictsExpressCopyrightGrant({value:wrong},[grant]),true);
});
test("do not mistake the substring in безотзывную for отзываемую",()=>{
 const correct="Предоставляется безотзывная авторская лицензия.";
 assert.equal(correctCopyrightGrantPolarity(correct,grant),correct);
 assert.equal(contradictsExpressCopyrightGrant({value:correct},[grant]),false);
});
test("do not invent a correction without primary-source proof",()=>{
 assert.equal(correctCopyrightGrantPolarity(wrong,"Licensor may distribute the Work."),wrong);
 assert.equal(contradictsExpressCopyrightGrant({value:wrong},["May distribute the work."]),false);
});
test("do not overwrite patent-specific litigation language",()=>{
 const patent="Патентная лицензия отзываемая при определённом судебном споре.";
 assert.equal(correctCopyrightGrantPolarity(patent,grant),patent);
 assert.equal(contradictsExpressCopyrightGrant({value:patent},[grant]),false);
});
