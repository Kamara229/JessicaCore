import {registerTool} from "./toolRegistry.js";
export function multiplyIntegers({a,b}={}) {
  if(!/^[+-]?\d{1,15}$/.test(String(a)) || !/^[+-]?\d{1,15}$/.test(String(b))) return {success:false,text:"Неверные числа"};
  const x=BigInt(a), y=BigInt(b);
  return {success:true,data:{product:String(x*y)},text:`${x} × ${y} = ${x*y}`};
}
registerTool({name:"integer_multiply",description:"Exact integer product",arguments:{a:"First integer",b:"Second integer"},execute:multiplyIntegers});