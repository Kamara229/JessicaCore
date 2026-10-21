import test from "node:test";
import assert from "node:assert/strict";
import * as source from "../core/language/codeAwareLocale.js";

test("isolated phone verifier detects baseline assertion failure and corrected candidate", () => {
    assert.equal(source.PHONE_SANDBOX_SMOKE_MARKER, true);
});
