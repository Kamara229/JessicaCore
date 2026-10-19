/*
 * An AI may select a legal clause from another source in a batch.
 * Downgrade JUST that cell to unsupported ([]) rather than discard
 * correctly resolved evidence for other cells. Never cross-attribute.
 */
export function validateCellClauseRefs(refs,cell,lookup,criterionType){
 const quotes=[],seen=new Set();
 for(const id of refs){
   if(!/^E\d+$/.test(id)||seen.has(id))
     return {valid:false,reason:"Повторный или некорректный ID выдержки"};
   seen.add(id);
   const passage=lookup.get(id);
   if(!passage || passage.source!==cell.source ||
       passage.criterion!==criterionType(cell.criterion))
     return {valid:true,quotes:[],misattributed:true};
   quotes.push(passage.quote);
 }
 return {valid:true,quotes,misattributed:false};
}
