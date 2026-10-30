// A narrowly bounded arithmetic chain; never swallow another operation.
const NUM="(\\(?[+-]?\\d{1,15}\\)?)";
const PATTERN=new RegExp(
  "^\\s*(?:вычисли|посчитай|рассчитай)\\s+"+NUM+
  "\\s*[×*xх]\\s*"+NUM+"\\s+и\\s+прибавь\\s+"+NUM+
  "\\s*\\.?\\s*(покажи\\s+расч[её]т\\s*\\.?)?\\s*$","iu"
);
function operand(input){
  const value=String(input||"");
  if(value.includes("(")!==value.includes(")"))return null;
  const number=value.replace(/[()]/gu,"");
  return /^[+-]?\d{1,15}$/u.test(number)?number:null;
}
export function parseIntegerProductAddTask(task){
  if(typeof task!=="string"||task.length>300)return null;
  const match=PATTERN.exec(task);
  if(!match)return null;
  const a=operand(match[1]),b=operand(match[2]),c=operand(match[3]);
  if(a===null||b===null||c===null)return null;
  return {a,b,c,showWork:Boolean(match[4])};
}
