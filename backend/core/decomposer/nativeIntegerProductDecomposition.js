import {parseIntegerProductTask} from "../native/integerMultiplicationIntent.js";

// A single explicit integer product is one complete semantic task.
// Decomposition adds no information and must not spend AI quota.
export function nativeIntegerProductDecomposition(task,memoryContext,memoryHints=[]) {
  if (!parseIntegerProductTask(task)) return null;
  return {
    handled:true,
    source:"native-integer-product",
    decomposition:{
      originalTask:task,
      isComplex:false,
      memoryContext,
      subtasks:[{
        id:1,
        text:task,
        dependsOn:[],
        contributesToFinalAnswer:true,
        memoryHints
      }]
    }
  };
}
