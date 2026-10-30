/** Fetch only the fixed, original GNU GPLv3 text without HTML/snippet ranking. */
const URL="https://www.gnu.org/licenses/gpl-3.0.txt";
export async function fetchCanonicalGpl3Text(url,request=globalThis.fetch){
 if(url!==URL)return null;
 try{
  const response=await request(URL,{method:"GET",redirect:"error",
   headers:{Accept:"text/plain","Accept-Language":"en"},
   signal:AbortSignal.timeout(12500)});
  if(!response?.ok)return null;
  const type=String(response.headers?.get?.("content-type")||"").toLowerCase();
  if(type && !type.includes("text/plain"))return null;
  const content=await response.text();
  if(content.length<20000||content.length>50000)return null;
  const text=content.replace(/\s+/gu," ").toLowerCase();
  const required=[
   "gnu general public license", "version 3, 29 june 2007",
   "4. conveying verbatim copies",
   "you may convey verbatim copies of the program's source code as you receive it",
   "conspicuously and appropriately publish on each copy an appropriate copyright notice",
   "keep intact all notices stating that this license",
   "keep intact all notices of the absence of any warranty",
   "give all recipients a copy of this license along with the program",
   "5. conveying modified source versions",
   "6. conveying non-source forms",
   "11. patents","end of terms and conditions"
  ];
  if(!required.every(x=>text.includes(x)))return null;
  return {success:true,text:"",data:{
   title:"GNU General Public License v3 - official original",
   url:URL,content,truncated:false,
   originalLength:content.length,returnedLength:content.length
  }};
 }catch{return null;}
}
