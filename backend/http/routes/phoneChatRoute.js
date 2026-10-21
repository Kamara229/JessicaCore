import {requireJessicaAuthorization} from "../auth/jessicaAuth.js";
import {validCodexOwnerKey, codexConfiguration} from "../../core/codex/codexSessionService.js";
import {preparePhoneChat, submitPhoneChat, readPhoneChat, stopPhoneChat} from "../../core/codex/phoneChatService.js";

export function registerPhoneChatRoutes(app) {
    const owner = (req, res, next) => {
        if (!validCodexOwnerKey(req.body?.ownerKey)) return res.status(403).json({reason: "owner-key-required"});
        next();
    };
    const handle = fn => async (req, res) => {
        try { res.set("Cache-Control", "no-store").json(await fn(req.body || {})); }
        catch (error) {
            // Errors are fixed policy/API codes; never echo responses, code or credentials.
            const reason = /^[a-z0-9-]+$/.test(error.message) ? error.message : "phone-operation-failed";
            res.status(409).set("Cache-Control", "no-store").json({reason});
        }
    };
    app.post("/api/codex/phone/prepare", requireJessicaAuthorization, owner, handle(body => {
        if (!codexConfiguration().ready) throw Error("github-owner-setup-required");
        return preparePhoneChat(body.goal, body.autoApply === true);
    }));
    app.post("/api/codex/phone/submit", requireJessicaAuthorization, owner,
        handle(body => submitPhoneChat(body.id, body.reply)));
    app.post("/api/codex/phone/status", requireJessicaAuthorization, owner,
        handle(body => readPhoneChat(body.id)));
    app.post("/api/codex/phone/stop", requireJessicaAuthorization, owner,
        handle(body => stopPhoneChat(body.id)));
}
