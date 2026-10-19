import test from "node:test";
import assert from "node:assert/strict";
import {unsupportedAbsolutePermission as guard} from "../core/validator/claim/comparisonAbsolutePermissionGuard.js";
test("permission does not prove lack of conditions",()=>{
 const quoted=["A copyright license to reproduce and prepare Derivative Works"];
 assert.equal(guard({value:"Можно изменять без ограничений"},quoted),true);
 assert.equal(guard({value:"Можно изменять при соблюдении условий"},quoted),false);
});
