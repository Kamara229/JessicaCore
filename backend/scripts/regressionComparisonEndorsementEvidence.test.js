import test from "node:test";
import assert from "node:assert/strict";
import {unsupportedEndorsementClaim} from "../core/validator/claim/comparisonEndorsementEvidence.js";
import {removeUnsupportedPromotion} from "../core/composer/comparisonUnsupportedPromotion.js";

const source="This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor, except as required for reasonable and customary use in describing the origin of the Work.";
const claim="Использование торговых марок не разрешено за пределами исключения; запрещено заявлять, что лицензия одобрена организацией, если это не так. «This License does not grant permission to use the trade names»";
test("trademark restriction is not independent approval prohibition",()=>{
 assert.equal(unsupportedEndorsementClaim(claim,[source]),true);
 const normalized=removeUnsupportedPromotion(claim,source);
 assert.doesNotMatch(normalized,/запрещено заявлять/iu);
 assert.match(normalized,/торговых марок/iu);
 assert.equal(unsupportedEndorsementClaim(normalized,[source]),false);
});
test("do not remove a restriction when original explicitly addresses endorsements",()=>{
 const proof=source+" Neither the name of the author may be used to endorse products without permission.";
 assert.equal(removeUnsupportedPromotion(claim,proof),claim);
});
