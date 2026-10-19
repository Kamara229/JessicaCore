import {buildComparisonClauseIndex} from "./comparisonClauseIndex.js";
/*
 * Positive evidence only. If a source already contains an operative
 * provision for a requested criterion, an "unable to verify" cell needs
 * a repair. The inverse (no indexed clause) NEVER proves its absence.
 */
export function indexExplicitComparisonTopics(sources) {
 const index=buildComparisonClauseIndex({results:sources})||[];
 const bySource=new Map();
 for(const clause of index) {
  const key=Number(clause.source);
  if(!bySource.has(key))bySource.set(key,new Set());
  bySource.get(key).add(clause.criterion);
 }
 return bySource;
}
export function hasExplicitComparisonTopic(index,source,criterion){
 return index.get(source)?.has(criterion)===true;
}
