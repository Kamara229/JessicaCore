/** Structural data handling is a base capability, separate from Experience. */
export const DATA_FORMATS=Object.freeze([
 {id:"json",ext:["json","jsonl"],aliases:["json","jsonl"]},
 {id:"csv",ext:["csv","tsv"],aliases:["csv","tsv"]},
 {id:"xml",ext:["xml","xsd","xsl"],aliases:["xml","xsd"]},
 {id:"spreadsheet",ext:["xlsx","xls","ods"],aliases:["excel","xlsx","spreadsheet","таблиц"]},
 {id:"sql",ext:["sql"],aliases:["sql","postgresql","mysql","sqlite","база данных"]},
 {id:"yaml",ext:["yaml","yml"],aliases:["yaml","yml"]},
 {id:"toml",ext:["toml"],aliases:["toml"]},
 {id:"parquet",ext:["parquet"],aliases:["parquet"]},
 {id:"protobuf",ext:["proto"],aliases:["protobuf","protocol buffers"]},
 {id:"api",ext:[],aliases:["rest api","graphql","openapi"]}
]);
const OPERATION=/(?:разбер|проанализир|проверь|валидир|сравни|преобраз|конвертир|структур|пол[яе]|колонк|строк|таблиц|сумм|объедин|очист|дубликат|фильтр|сортир|агрегир|схем|запрос|рассч|данны|парс|прочитай|покажи|analy[sz]e|validate|parse|convert|transform|schema|data|columns|rows|filter|sort|aggregate|join|duplicates|query)/iu;
export function analyzeDataTask(task,files=[]) {
 const text=String(task||"");
 const names=(Array.isArray(files)?files:[]).map(f=>String(typeof f==="string"?f:f?.name||"").toLowerCase());
 const codeFence=/\x60{3}(json|csv|xml|yaml|sql|toml|tsv)(?:\s|\n)/iu.test(text);
 const matched=DATA_FORMATS.filter(f=>f.aliases.some(a=>new RegExp("(^|[^\\p{L}\\p{N}_])"+a+"(?=$|[^\\p{L}\\p{N}_])","iu").test(text))||names.some(n=>f.ext.some(e=>n.endsWith("."+e))));
 const inputIsData=names.some(n=>DATA_FORMATS.some(f=>f.ext.some(e=>n.endsWith("."+e))));
 const relevant=codeFence||(matched.length>0&&OPERATION.test(text))||(inputIsData&&OPERATION.test(text));
 return Object.freeze({relevant, formats:Object.freeze(matched.map(x=>x.id)),hasDataFiles:inputIsData,executed:false});
}
