/*
 * The user must see real source URLs, not internal evidence IDs.
 * Derive links exclusively from fetched documents. Never invent a URL
 * from a model answer or infer that adding a link proves a legal claim.
 */
function sourceUrls(run) {
  const items=(run?.results||[]).filter(x=>
    x?.tool==="web_fetch"&&x?.success===true&&x?.data?.url);
  if(items.length!==2)return null;
  const urls=items.map(item=>{
    try {
      const u=new URL(String(item.data.url));
      return ["https:","http:"].includes(u.protocol)?u.href:null;
    }catch{return null;}
  });
  return urls.every(Boolean)&&urls[0]!==urls[1]?urls:null;
}
export function attributeComparisonSources(answer,run) {
  const urls=sourceUrls(run);
  if(!urls)return {text:String(answer||""),changes:0};
  let changes=0;
  let output=String(answer||"").replace(
    /【\s*S([12])[-‑][^】]{0,75}】/giu,
    (full,sourceIndex)=>{
      const url=urls[Number(sourceIndex)-1];
      if(!url)return full;
      changes++;
      return " [Официальный источник "+sourceIndex+"]("+url+")";
    }
  );
  // Cite the actual fetched texts even if the Composer forgot a footer.
  if(!urls.every(url=>output.includes(url))){
    output=output.trimEnd()+"\n\nОфициальные тексты: "+
      urls.map((url,i)=>"[Источник "+(i+1)+"]("+url+")").join("; ")+".";
    changes++;
  }
  return {text:output,changes};
}
