import {parseLatestOrderSqlTask} from "../native/latestOrderSqlIntent.js";
export function nativeLatestOrderSqlDecomposition(task,memoryContext,memoryHints=[]){
 if(!parseLatestOrderSqlTask(task))return null;
 return {handled:true,source:"native-latest-order-sql",decomposition:{
 originalTask:task,isComplex:false,memoryContext,
 subtasks:[{id:1,text:task,dependsOn:[],contributesToFinalAnswer:true,memoryHints}]
 }};
}
