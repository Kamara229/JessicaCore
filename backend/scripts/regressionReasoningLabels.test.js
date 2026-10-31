import test from "node:test";
import assert from "node:assert/strict";
import {getComposerInstructions} from "../core/composer/composerPrompt.js";
import {buildAIResultValidatorMessages} from "../core/validator/aiResult/aiResultPrompt.js";
import {getValidatorMaxCompletionTokens} from "../ai/validatorClient.js";

test("explicit box labels remain grounded throughout answer and validation prompts",()=>{
 const task="Коробки: яблоки, груши, смесь. Все ярлыки неверны. Из какой взять один фрукт, чтобы определить все коробки? Почему?";
 const composer=getComposerInstructions(task);
 assert.match(composer,/Не заменяй конкретные названия из задачи/iu);
 assert.match(composer,/Если пользователь спрашивает «почему»/iu);
 const answer="Нужно взять фрукт из коробки «Яблоки и Апельсины».";
 const messages=buildAIResultValidatorMessages(task,{evidence:{mode:"none"}},
    {success:true,results:[]},{success:true,text:answer});
 assert.ok(messages[1].content.includes(task));
 assert.ok(messages[1].content.includes(answer));
 assert.match(messages[0].content,/подмена одного предмета другим/iu);
 assert.match(messages[0].content,/если пользователь просит объяснить почему/iu);
 assert.equal(getValidatorMaxCompletionTokens(),600);
});
