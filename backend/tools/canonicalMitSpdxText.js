/** Verify the MIT original against the official SPDX license-list-data text. */
const RAW="https://raw.githubusercontent.com/spdx/license-list-data/main/text/MIT.txt";
const REFERENCE="https://spdx.org/licenses/MIT.html";
export const SPDX_MIT_RAW_URL=RAW;
export const SPDX_MIT_REFERENCE_URL=REFERENCE;

export async function fetchSpdxMitText(request=globalThis.fetch) {
 try {
  const response=await request(RAW,{method:"GET",redirect:"error",
   headers:{Accept:"text/plain","Accept-Language":"en"},
   signal:AbortSignal.timeout(9000)});
  if(!response?.ok)return null;
  const contentType=String(response.headers?.get?.("content-type")||"").toLowerCase();
  if(contentType && !contentType.includes("text/plain"))return null;
  const body=await response.text();
  if(body.length<750 || body.length>10000)return null;
  const flat=body.replace(/\s+/gu," ").toLowerCase();
  const checks=["mit license","permission is hereby granted, free of charge",
   "including without limitation the rights to use, copy, modify, merge, publish, distribute",
   "the above copyright notice and this permission notice shall be included in all copies or substantial portions of the software.",
   'the software is provided "as is"',
   "in no event shall the authors or copyright holders be liable",
   "other dealings in the software."];
  if(!checks.every(x=>flat.includes(x)))return null;
  return {success:true,text:"",data:{
   title:"MIT License - SPDX maintained original",url:RAW,
   referenceUrl:REFERENCE,content:body,truncated:false,
   originalLength:body.length,returnedLength:body.length
  }};
 }catch{return null;}
}
