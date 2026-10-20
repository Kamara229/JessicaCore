import {requireJessicaAuthorization} from "../auth/jessicaAuth.js";
import {replaceChatDebugArchive} from "../../storage/chatDebugArchive.js";

export function registerChatDebugArchiveRoute(app){
    app.post("/api/chat/debug/sync",requireJessicaAuthorization,async(req,res)=>{
        try {
            const result=await replaceChatDebugArchive(req.body);
            if(result.invalid)return res.status(400).json({success:false,reason:result.reason});
            if(!result.saved)return res.status(503).json({success:false,reason:result.reason});
            return res.set("Cache-Control","no-store").json({success:true,
                threads:result.threads,messages:result.messages});
        }catch {
            return res.status(503).json({success:false,reason:"chat-diagnostic-unavailable"});
        }
    });
}
