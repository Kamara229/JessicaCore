import test from "node:test";
import assert from "node:assert/strict";
import {extractPatentClauseExcerpts} from "../core/validator/claim/patentClauseExcerpts.js";
import {buildComparisonEvidenceGrid,renderComparisonEvidenceGrid} from "../core/composer/comparisonEvidenceGrid.js";

const apache=[
 "Apache License Version 2.0. 3. Grant of Patent License.",
 "Each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contributions alone.",
 "If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed."
].join(" ");
const bsd="Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met. Redistributions of source code must retain the above copyright notice.";

test("source excerpts preserve the limiting condition of patent termination",()=>{
 const excerpts=extractPatentClauseExcerpts(apache,"патентные условия");
 assert.ok(excerpts.some(x=>x.includes("alleging that the Work or a Contribution incorporated within the Work")));
 const run={results:[
  {tool:"web_fetch",success:true,data:{url:"https://www.apache.org/licenses/LICENSE-2.0.txt",content:apache}},
  {tool:"web_fetch",success:true,data:{url:"https://opensource.org/license/bsd-3-clause",content:bsd}}
 ]};
 const context=renderComparisonEvidenceGrid(buildComparisonEvidenceGrid("Сравни патентные условия",run));
 assert.match(context,/alleging that the Work or a Contribution/);
 assert.match(context,/patent infringement/);
 assert.match(context,/shall terminate/);
 assert.doesNotMatch(context,/патентных прав нет/);
});
