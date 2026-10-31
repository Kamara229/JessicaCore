import {parseSafeAccountTransferTask} from "../native/safeAccountTransferIntent.js";
export function nativeSafeAccountTransferDecomposition(task,memoryContext,memoryHints=[]){
 if(!parseSafeAccountTransferTask(task))return null;
 return {handled:true,source:"native-account-transfer-sql",decomposition:{
  originalTask:task,isComplex:false,memoryContext,
  subtasks:[{id:1,text:task,dependsOn:[],contributesToFinalAnswer:true,memoryHints}]
 }};
}
