import { randomUUID } from "node:crypto";
import { getSupabaseClient, isSupabaseConfigured } from "./supabaseClient.js";

/*
 * One private Supabase diagnostic snapshot per /api/solve request.
 * Never store bearer tokens or attached document text. This mirror is
 * unrelated to the Experience / Learning pipeline.
 *
 * The Android app sends a bounded chat excerpt (historical messages
 * are potentially truncated to 800 characters); the current request
 * and HTTP response are captured verbatim.
 */
export function buildChatDebugSnapshot({task,conversation,attachments,result}={}) {
    const history=(Array.isArray(conversation)?conversation:[])
        .filter(item=>item && (item.role==="user"||item.role==="assistant")
            && typeof item.content==="string"&&item.content.trim())
        .slice(-8)
        .map(item=>({
            role:item.role,
            text:item.content,
            origin:"android_context_excerpt"
        }));
    const names=(Array.isArray(attachments)?attachments:[])
        .filter(a=>a && typeof a.name==="string")
        .slice(0,3)
        .map(a=>a.name.slice(0,180));
    const current={
        role:"user",
        text:String(task??""),
        origin:"android_current_request",
        attachments:names
    };
    const reply={
        role:"assistant",
        text:String(result?.text??""),
        origin:"backend_http_response"
    };
    return {
        slot:"latest",
        request_id:randomUUID(),
        captured_at:new Date().toISOString(),
        messages:[...history,current,reply].slice(-10),
        result_success:result?.success===true,
        note:"Предыдущие сообщения — сокращённый контекст Android (до 800 символов на сообщение). Последний запрос и ответ сервера сохранены полностью."
    };
}

export async function saveChatDebugSnapshot(input) {
    if(String(process.env.JESSICA_CHAT_DEBUG_ENABLED||"").toLowerCase()!=="true") {
        return {saved:false,reason:"disabled"};
    }
    if(!isSupabaseConfigured()) return {saved:false,reason:"supabase-not-configured"};

    const snapshot=buildChatDebugSnapshot(input);
    const abort=new AbortController();
    const timer=setTimeout(()=>abort.abort(),2500);
    try {
        const {error}=await getSupabaseClient()
            .from("jessica_debug_current_chat")
            .upsert(snapshot,{onConflict:"slot"})
            .abortSignal(abort.signal);
        if(error) {
            console.warn("Jessica private chat mirror write failed:", error.code||"db-error");
            return {saved:false,reason:"db-error"};
        }
        console.log("Jessica private chat mirror updated:", {
            requestId:snapshot.request_id,
            messages:snapshot.messages.length,
            successfulReply:snapshot.result_success
        });
        return {saved:true,count:snapshot.messages.length};
    } catch(error) {
        console.warn("Jessica private chat mirror unavailable:",error?.name||"error");
        return {saved:false,reason:"unavailable"};
    } finally {
        clearTimeout(timer);
    }
}
