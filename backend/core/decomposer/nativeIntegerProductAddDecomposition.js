import {parseIntegerProductAddTask} from "../native/integerProductAddIntent.js";

export function nativeIntegerProductAddDecomposition(task,memoryContext,memoryHints=[]){
  if(!parseIntegerProductAddTask(task))return null;
  return {handled:true,source:"native-integer-product-add",decomposition:{
    originalTask:task,isComplex:false,memoryContext,
    subtasks:[{id:1,text:task,dependsOn:[],
      contributesToFinalAnswer:true,memoryHints}]
  }};
}
