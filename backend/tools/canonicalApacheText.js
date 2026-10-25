/**
 * Retrieve one public, fixed-host plain-text document without a renderer.
 * The URI is not user-controlled. Any failure defers to existing fetch.
 */
const APACHE_TEXT="https://www.apache.org/licenses/LICENSE-2.0.txt";

export async function fetchCanonicalApacheText(url,request=globalThis.fetch){
    if(url!==APACHE_TEXT)return null;
    try{
        const response=await request(APACHE_TEXT,{
            method:"GET",redirect:"error",
            headers:{Accept:"text/plain","Accept-Language":"en"},
            signal:AbortSignal.timeout(18000)
        });
        if(!response?.ok)return null;
        const type=String(response.headers?.get?.("content-type")||"").toLowerCase();
        if(type && !type.includes("text/plain"))return null;
        const content=await response.text();
        if(content.length>30000 || content.length<6000 ||
           !/3\.\s*Grant of Patent License/u.test(content) ||
           !/4\.\s*Redistribution/u.test(content) ||
           !/If You institute patent litigation/u.test(content) ||
           !/shall terminate\s+as of the date such litigation is filed/u.test(content))
           return null;
        return {success:true,text:"",data:{
            title:"Apache License 2.0 - original text",
            url:APACHE_TEXT,content,truncated:false,
            originalLength:content.length,returnedLength:content.length
        }};
    }catch{return null;}
}
