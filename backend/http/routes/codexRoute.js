import {requireJessicaAuthorization} from "../auth/jessicaAuth.js";
import {codexConfiguration,validCodexOwnerKey,startCodexSession,
    readCodexSession,stopCodexSession,codexGoals,startCodexDay} from "../../core/codex/codexSessionService.js";
import {reviewCodexEngineering} from "../../core/codex/codexPeerReview.js";

/**
 * Privileged actions require a separate owner key; JESSICA_APP_TOKEN embedded
 * in APKs is not trusted as authorization to spend Codex credits or edit repos.
 */
export function registerCodexRoutes(app) {
    app.get("/api/codex/config",requireJessicaAuthorization,(_req,res)=>
        res.set("Cache-Control","no-store").json(codexConfiguration()));

    const onlyOwner=(req,res,next)=>{
        if(!validCodexOwnerKey(req.body?.ownerKey))return res.status(403)
            .json({success:false,reason:"owner-key-required"});
        next();
    };
    // Owner-only free daytime checks. Legacy paid Codex flow stays opt-in.
    app.post("/api/codex/day/start",requireJessicaAuthorization,onlyOwner,async(req,res)=>{
        const result=await startCodexDay(req.body?.goal||"auto");
        res.set("Cache-Control","no-store").status(result.status).json(result);
    });
    app.post("/api/codex/start",requireJessicaAuthorization,onlyOwner,async(req,res)=>{
        const result=await startCodexSession(req.body?.goal||"foundations");
        res.set("Cache-Control","no-store").status(result.status).json(result);
    });
    app.post("/api/codex/status",requireJessicaAuthorization,onlyOwner,async(req,res)=>{
        const result=await readCodexSession(req.body?.issue);
        res.set("Cache-Control","no-store").status(result.status).json(result);
    });
    app.post("/api/codex/stop",requireJessicaAuthorization,onlyOwner,async(req,res)=>{
        const result=await stopCodexSession(req.body?.issue);
        res.set("Cache-Control","no-store").status(result.status).json(result);
    });
    // For the trusted GitHub Actions runner. No user chat content is passed.
    app.post("/api/codex/peer-review",requireJessicaAuthorization,async(req,res)=>{
        const goal=String(req.body?.goal||"");
        if(!codexGoals().includes(goal) ||
            !["proposal","diff"].includes(req.body?.phase) ||
            typeof req.body?.content!=="string" ||
            req.body.content.length>16000)
            return res.status(400).json({success:false,reason:"invalid-review"});
        const result=await reviewCodexEngineering({
            goal,phase:req.body.phase,content:req.body.content});
        res.set("Cache-Control","no-store").status(result.success?200:503).json(result);
    });
}
