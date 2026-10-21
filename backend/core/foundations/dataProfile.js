/** Small deterministic source-data summaries; never persist private values in Skills. */
const MAX_FIELDS=32;
function errorProfile(format,name,error,truncated=false) {
    return {format,name,valid:false,partial:truncated,error:String(error||"unreadable").slice(0,150)};
}
function jsonFields(value,limit=MAX_FIELDS) {
    const fields=[],visited=new Set();
    function visit(current,prefix,depth) {
        if(fields.length>=limit||depth>3||!current||typeof current!=="object") return;
        const source=Array.isArray(current)?current.slice(0,2):Object.entries(current).slice(0,80);
        for(const item of source) {
            if(fields.length>=limit) break;
            const key=Array.isArray(current)?String(source.indexOf(item)):String(item[0]);
            const data=Array.isArray(current)?item:item[1];
            const field=prefix?prefix+"."+key:key;
            if(visited.has(field)) continue;
            visited.add(field);
            const type=data===null?"null":Array.isArray(data)?"array":typeof data;
            fields.push({name:field.slice(0,100),type});
            if(type==="object"||type==="array") visit(data,field,depth+1);
        }
    }
    visit(value,"",0);
    return fields;
}
export function profileJson(text,{name="data.json",truncated=false}={}) {
    let data;
    try{data=JSON.parse(String(text||""));}
    catch(error){return errorProfile("json",name,"JSON syntax error: "+error.message,truncated);}
    const shape=data===null?"null":Array.isArray(data)?"array":typeof data;
    return {format:"json",name,valid:true,partial:truncated,topLevelType:shape,
        topLevelItems:Array.isArray(data)?data.length:shape==="object"?Object.keys(data).length:null,
        fields:jsonFields(data)};
}
/** RFC4180-like strict CSV quote parsing. Strings are never auto-coerced. */
export function parseDelimited(text,delimiter=",",maxRows=1500) {
    if(typeof text!=="string"||!text.length)throw Error("No CSV text");
    const rows=[];let row=[],field="",quoted=false,closed=false,overflow=false;
    function pushField(){row.push(field);field="";closed=false;}
    function pushRow(){pushField();rows.push(row);row=[];if(rows.length>maxRows)overflow=true;}
    for(let i=0;i<text.length;i++){
        const ch=text[i];
        if(quoted) {
            if(ch==='"') {
                if(text[i+1]==='"'){field+='"';i++;}else{quoted=false;closed=true;}
            }else{field+=ch;}
        }else if(ch==='"'&&field===" "){
            throw Error("Invalid CSV quote placement");
        }else if(ch==='"'&&field===""){quoted=true;}
        else if(ch==='"'){throw Error("Invalid CSV quote placement");}
        else if(ch===delimiter){pushField();}
        else if(ch==="\n"||ch==="\r"){
            if(ch==="\r"&&text[i+1]==="\n")i++;
            pushRow();
            if(rows.length>maxRows)break;
        }else if(closed&&ch!==" "){throw Error("Unexpected character after closing quote");}
        else if(!closed){field+=ch;}
    }
    if(quoted)throw Error("Unclosed quoted CSV field");
    if(!overflow&&(field.length||row.length))pushRow();
    return {rows:rows.slice(0,maxRows),overflow};
}
export function profileCsv(text,{name="data.csv",truncated=false}={}) {
    const header=String(text||"").split(/\r?\n/,1)[0]||"";
    const delimiter=header.includes("\t")?"\t":
        header.includes(";")&&header.split(";").length>header.split(",").length?";":",";
    try {
        const parsed=parseDelimited(String(text||""),delimiter);
        const rows=parsed.rows;
        const headers=rows[0]||[];
        const samples=rows.slice(1);
        const mismatch=samples.filter(r=>r.length!==headers.length).length;
        return {format:"csv",name,valid:headers.length>0,partial:truncated||parsed.overflow,
            delimiter:delimiter==="\t"?"tab":delimiter,
            columns:headers.slice(0,MAX_FIELDS).map(x=>x.slice(0,100)),
            columnsOmitted:Math.max(0,headers.length-MAX_FIELDS),
            inspectedRows:samples.length,
            mismatchedRows:mismatch,
            totalRowsKnown:!(truncated||parsed.overflow)};
    }catch(error){return errorProfile("csv",name,error.message,truncated);}
}
export function profileStructuredAttachments(attachments) {
    if(!Array.isArray(attachments))return [];
    return attachments.slice(0,3).flatMap(file=>{
        const name=String(file?.name||"");
        const lower=name.toLowerCase();
        const text=String(file?.text||"");
        if(lower.endsWith(".json"))return [profileJson(text,{name,truncated:file?.truncated===true})];
        if(lower.endsWith(".csv")||lower.endsWith(".tsv"))
            return [profileCsv(text,{name,truncated:file?.truncated===true})];
        return [];
    });
}
