import {requestComposerAI} from "../../ai/composerClient.js";
import {planFocusedComparisonRepair} from "./focusedComparisonPlan.js";
import {patchFocusedComparison} from "./focusedComparisonPatch.js";

/* Optional surgical repair. A fresh independent Validator must approve
 * every changed statement; this function never certifies legal meaning.
 */
const instructions=[
 "Ты исправляешь ТОЛЬКО неподтверждённые ячейки сравнения двух первоисточников.",
 "Для каждого ID составь ОДИН короткий юридически содержательный вывод на русском.",
 "Используй только цитаты, приложенные к этому ID и его SOURCE.",
 "Различай разрешение, обязанность, условие и запрет.",
 "Не добавляй отсутствующие в выдержках факты, названия пунктов, гарантии и дополнительные ограничения.",
 "Обязанность включить копию уведомлений НЕ означает обязанность сохранить исходный файл.",
 "Не пиши «без ограничений», если текст устанавливает условия.",
 "Если недостаточно данных, используй «Не удалось подтвердить по прочитанному тексту».",
 "Ответ — только JSON: {\"c\":[[1,\"Исправленная формулировка\"],[2,\"Исправленная формулировка\"]]}.",
 "Не меняй исходную таблицу целиком, ответь только для полученных ID."
].join("\n");
export async function tryFocusedComparisonRepair(task,run,repair){
 const plan=planFocusedComparisonRepair(task,run,repair);
 if(!plan)return null;
 try {
   const response=await requestComposerAI({
     instructions,
     input:"Задача: "+task+"\nТребуют исправления ТОЛЬКО:\n"+
       JSON.stringify(plan.cells)+"\nПиши кратко, сохрани условия норм.",
     pairedSources:true
   });
   const content=response?.choices?.[0]?.message?.content||"";
   const result=patchFocusedComparison(plan,content);
   if(!result.ok)return null;
   console.log("Jessica Focused Comparison Repair:",{
     candidateCells:plan.cells.length,patched:result.changed
   });
   return {success:true,text:result.text,source:"focused-composer"};
 }catch(error){
   console.warn("Jessica Focused Comparison Repair unavailable:",
     error?.status||error?.name||"error");
   return null;
 }
}
