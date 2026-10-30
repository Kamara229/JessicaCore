import {registerTool} from "./toolRegistry.js";
export function multiplyIntegers({a,b,showWork=false}={}) {
  if(!/^[+-]?\d{1,15}$/.test(String(a)) || !/^[+-]?\d{1,15}$/.test(String(b))) return {success:false,text:"Неверные числа"};
  const x=BigInt(a), y=BigInt(b), result=x*y;
  const chunks=String(y<0n?-y:y).split("").map((digit,i,all)=>x*BigInt(digit)*10n**BigInt(all.length-1-i)*(y<0n?-1n:1n));
  const text=showWork ? "По разрядам: "+chunks.join(" + ")+" = "+result+"\nОтвет: "+x+" × "+y+" = "+result : x+" × "+y+" = "+result;
  return {success:true,data:{product:String(result)},text};
}
registerTool({name:"integer_multiply",description:"Exact integer product",arguments:{a:"First integer",b:"Second integer",showWork:"Explain by place value"},execute:multiplyIntegers});