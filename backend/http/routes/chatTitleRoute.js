import {requireJessicaAuthorization} from "../auth/jessicaAuth.js";
import {suggestChatTitle} from "../../core/chat/chatTitleService.js";

export function registerChatTitleRoute(app){
    app.post("/api/chat/title",requireJessicaAuthorization,async(req,res)=>{
        const task=typeof req.body?.task==="string"?req.body.task.trim():"";
        if(!task||task.length>1800) return res.status(400).json({success:false});
        const answer=typeof req.body?.answer==="string"?req.body.answer.slice(0,1200):"";
        const title=await suggestChatTitle({task,answer});
        return res.set("Cache-Control","no-store").json({success:true,title});
    });
}
