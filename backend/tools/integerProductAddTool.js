import {registerTool} from "./toolRegistry.js";
import {multiplyIntegers} from "./integerMultiplicationTool.js";

export function multiplyAndAddIntegers({a,b,c,showWork=false}={}){
  if(![a,b,c].every(value=>/^[+-]?\d{1,15}$/u.test(String(value))))
    return {success:false,text:"Неверные целочисленные операнды"};
  const product=multiplyIntegers({a,b});
  if(product.success!==true)return product;
  const sum=BigInt(product.data.product)+BigInt(c);
  const text=showWork
    ? "Расчёт: "+a+" × "+b+" = "+product.data.product+
      "; "+product.data.product+" + ("+c+") = "+sum+
      "\nОтвет: "+sum
    : "Ответ: "+a+" × "+b+" + ("+c+") = "+sum;
  return {success:true,data:{product:product.data.product,sum:String(sum)},text};
}
registerTool({name:"integer_multiply_add",
  description:"Exact integer multiplication followed by addition",
  arguments:{a:"First integer",b:"Second integer",c:"Integer to add",showWork:"Show both operations"},
  execute:multiplyAndAddIntegers});
