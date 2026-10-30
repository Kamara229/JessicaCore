import test from "node:test";
import assert from "node:assert/strict";
import {fetchCanonicalMitText} from "../tools/canonicalMitText.js";
test("Reject other URL", async()=>{assert.equal(await fetchCanonicalMitText("https://example.org/other"),null);});
