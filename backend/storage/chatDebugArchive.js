import {getSupabaseClient,isSupabaseConfigured} from "./supabaseClient.js";

const UUID=/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const MAX_BYTES=850_000;
/* This is an owner-only diagnostic mirror, not cloud history or long-term Experience.
 * Replacing a single snapshot means a deleted chat disappears from this mirror.
 * Attachment content and authentication secrets are never collected.
 */
export function validateChatArchive(input) {
    if(!input||!Array.isArray(input.threads)||input.threads.length>100)
        return {valid:false,reason:"Некорректный список чатов"};
    const ids=new Set(),out=[];
    let messages=0;
    for(const thread of input.threads){
        if(typeof thread?.id!=="string"||!UUID.test(thread.id)||ids.has(thread.id)
            ||typeof thread.title!=="string"||thread.title.length>100
            ||!Array.isArray(thread.messages)||thread.messages.length>3000)
            return {valid:false,reason:"Некорректные данные чата"};
        ids.add(thread.id);
        const clean=[];
        for(const message of thread.messages){
            if(typeof message?.text!=="string"||message.text.length>20000
                ||typeof message?.isUser!=="boolean"||!Number.isFinite(message?.timestamp))
                return {valid:false,reason:"Некорректное сообщение чата"};
            clean.push({text:message.text,isUser:message.isUser,
                timestamp:Math.trunc(message.timestamp)});
            messages++;
        }
        out.push({id:thread.id,title:thread.title,
            updatedAt:Number.isFinite(thread.updatedAt)?Math.trunc(thread.updatedAt):0,
            messages:clean});
    }
    const active=typeof input.activeThreadId==="string"&&ids.has(input.activeThreadId)
        ?input.activeThreadId:null;
    const row={slot:"device",threads:out,active_thread_id:active,
        thread_count:out.length,message_count:messages,synced_at:new Date().toISOString()};
    if(Buffer.byteLength(JSON.stringify(row),"utf8")>MAX_BYTES)
        return {valid:false,reason:"Архив превышает лимит диагностики; данные не были усечены"};
    return {valid:true,row};
}
export async function replaceChatDebugArchive(body) {
    if(String(process.env.JESSICA_CHAT_DEBUG_ENABLED||"").toLowerCase()!=="true")
        return {saved:false,reason:"disabled"};
    if(!isSupabaseConfigured())return {saved:false,reason:"supabase-not-configured"};
    const checked=validateChatArchive(body);
    if(!checked.valid)return {saved:false,reason:checked.reason,invalid:true};
    const {error}=await getSupabaseClient()
        .from("jessica_debug_chat_archive")
        .upsert(checked.row,{onConflict:"slot"});
    if(error)return {saved:false,reason:"database-error"};
    console.log("Jessica full chat diagnostic synchronized:",{
        threads:checked.row.thread_count,messages:checked.row.message_count
    });
    return {saved:true,threads:checked.row.thread_count,messages:checked.row.message_count};
}
