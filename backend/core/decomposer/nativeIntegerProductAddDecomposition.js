import {parseSequencePriceTask} from "../native/sequencePriceIntent.js";
import {parseIntegerProductAddTask} from "../native/integerProductAddIntent.js";

export function nativeIntegerProductAddDecomposition(task,memoryContext,memoryHints=[]){
  const price=parseSequencePriceTask(task);
  if(!price && !parseIntegerProductAddTask(task))return null;
  return {handled:true,source:price?"native-sequential-price":"native-integer-product-add",decomposition:{
    originalTask:task,isComplex:false,memoryContext,
    subtasks:[{id:1,text:task,dependsOn:[],
      contributesToFinalAnswer:true,memoryHints}]
  }};
}
