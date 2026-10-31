import {parseBridgeCrossingTask} from "../native/bridgeCrossingIntent.js";
// Keep a fully specified bridge puzzle intact before any AI paraphrasing.
export function nativeBridgeDecomposition(task,memoryContext,memoryHints=[]){
  if(!parseBridgeCrossingTask(task))return null;
  return {handled:true,source:"native-bridge",decomposition:{
    originalTask:task,isComplex:false,memoryContext,
    subtasks:[{id:1,text:task,dependsOn:[],contributesToFinalAnswer:true,memoryHints}]
  }};
}
