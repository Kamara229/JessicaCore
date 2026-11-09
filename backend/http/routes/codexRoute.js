import {requireJessicaAuthorization} from "../auth/jessicaAuth.js";
import {interpretRelayTurn,explicitJessicaTest} from "../../core/codex/relayInterpreter.js";
import {compactJessicaTest} from "../../core/codex/relayCompactTest.js";
import {numberedBenchmarkTest} from "../../core/codex/relayNumberedBenchmarkTest.js";
import {safeCalibriPassiveHandoff} from "../../core/codex/relayCalibriPassiveSafe.js";
import {recordCodexPhase} from "../../core/codex/relayTelemetry.js";
import {interpretationFailureHandoff} from "../../core/codex/relayFailureHandoff.js";

/** Codex is a chat participant. Repo-writing, paid and day routes are no longer registered. */
export function registerCodexRoutes(app) {
    // No sensitive ChatGPT messages or user code are logged by this route.
    app.post("/api/codex/relay/phase", requireJessicaAuthorization, (req,res) => {
        res.set("Cache-Control","no-store");
        try {
            recordCodexPhase(req.body);
            return res.status(204).end();
        } catch {
            return res.status(400).json({success:false,reason:"invalid-relay-phase"});
        }
    });
    app.post("/api/codex/relay/interpret", requireJessicaAuthorization, async (req,res) => {
        res.set("Cache-Control","no-store");
        try {
            // Strict no-test relay-only CALIBRI status: never spend provider
            // quota or confuse an idle instruction with a Jessica benchmark.
            const explicitTask=explicitJessicaTest(req.body?.message) ||
                compactJessicaTest(req.body?.message) ||
                numberedBenchmarkTest(req.body?.message);
            const passive=explicitTask ? null : safeCalibriPassiveHandoff(req.body?.message);
            const decision = passive ?? await interpretRelayTurn(req.body);
            res.json({success:true,...decision});
        } catch (error) {
            // An interpretation outage must NOT disable the already-running
            // Codex relay. Valid authenticated turns receive a safe non-test
            // chat continuation instead of HTTP 503 / phone-side pause.
            const message=req.body?.message;
            if (typeof message!=="string" || !message.trim() || message.length>50000)
                return res.status(400).json({success:false,reason:"invalid-message"});
            return res.json({success:true,...interpretationFailureHandoff(error)});
        }
    });
}
