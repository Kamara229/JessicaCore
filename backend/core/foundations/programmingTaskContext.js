import {PROGRAMMING_LANGUAGES} from "./programmingLanguageCatalog.js";

const CODING_INTENT=/(?:\b(?:code|coding|debug|refactor|compile|compiler|script|source code|programming|function|class|method|syntax|implement|unit test|write a function|review code|fix code)\b|(?<![\\p{L}\\p{N}])код(?:а|у|ом|е|ы|ов)?(?![\\p{L}\\p{N}])|скрипт|программир|исходник|скомпил|отлад|рефактор|реализуй|исправь код|напиши (?:функци|класс|программ|скрипт|код)|синтаксис|функци[яию]|компиляц|тесты для кода)/iu;
const FENCES = /\x60{3}\s*([a-z0-9+#._-]+)?\s*\n/giu;

function mentions(text,alias) {
    const escaped=alias.replaceAll("+","\\+").replaceAll(".","\\.");
    return new RegExp("(^|[^\\p{L}\\p{N}_])"+escaped+"(?=$|[^\\p{L}\\p{N}_])","iu").test(text);
}
export function detectProgrammingLanguages(task,files=[]) {
    const text=String(task||"");
    const names=(Array.isArray(files)?files:[]).map(item=>String(typeof item==="string"?item:item?.name||"").toLowerCase());
    const fenced=[...text.matchAll(FENCES)].map(match=>String(match[1]||"").toLowerCase()).filter(Boolean);
    const matches=[];
    for(const language of PROGRAMMING_LANGUAGES) {
        const filesMatch=names.some(name=>language.extensions.some(ext=>name.endsWith(ext)));
        const fenceMatch=fenced.some(tag=>tag===language.id||language.aliases.includes(tag));
        const named=language.aliases.some(alias=>alias.length>1&&mentions(text,alias));
        if(filesMatch||fenceMatch||named) matches.push(language);
    }
    return matches.slice(0,10);
}
export function analyzeProgrammingTask(task,files=[]) {
    const text=String(task||"").slice(0,24000);
    const names=(Array.isArray(files)?files:[]).map(item=>String(typeof item==="string"?item:item?.name||""));
    const languages=detectProgrammingLanguages(text,names);
    const fenced=Array.from(text.matchAll(FENCES));
    // Configuration and data extensions alone are not proof of a coding task.
    const dataOnly=new Set([".json",".jsonl",".yaml",".yml",".xml",".xsd",".xsl",".toml",".html",".htm",".css",".scss",".sass",".less",".tf",".tfvars",".hcl",".md",".txt",".csv"]);
    const codeFiles=names.some(name=>PROGRAMMING_LANGUAGES.some(lang=>
        lang.extensions.some(ext=>!dataOnly.has(ext)&&name.toLowerCase().endsWith(ext))));
    const syntax=/(?:^|\n)\s*(?:def |class |function |const |let |var |#include|import |package |fn |pub fn |SELECT |CREATE TABLE |<html|#!\/|using System|public static|func |int main\s*\()/im.test(text);
    const intended=CODING_INTENT.test(text);
    const relevant=(intended || fenced.length>0 || codeFiles || (syntax && languages.length>0));
    return Object.freeze({
        relevant,
        languages:Object.freeze(languages.map(item=>item.id)),
        names:Object.freeze(languages.map(item=>item.name)),
        hasCodeFence:fenced.length>0,
        codeFiles,
        // Context is descriptive. It NEVER indicates compiled or executed code.
        executed:false
    });
}
