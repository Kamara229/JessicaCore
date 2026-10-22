import {requireJessicaAuthorization} from "../auth/jessicaAuth.js";
import {interpretRelayTurn} from "../../core/codex/relayInterpreter.js";
import {recordCodexPhase} from "../../core/codex/relayTelemetry.js";

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
            const decision = await interpretRelayTurn(req.body);
            res.json({success:true,...decision});
        } catch {
            res.status(503).json({success:false,reason:"relay-understanding-unavailable",
                text:"Не удалось уверенно понять ответ ChatGPT. Переписка приостановлена."});
        }
    });
}
