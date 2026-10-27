import assert from "node:assert/strict";

// Read-only live relay check: a confirmed backend CI status never requires a
// quota-bound interpreter nor triggers a test in Jessica's ordinary chat.
export async function verifyCurrentBackendProgress(interpret){
  const payload={
    plan:"Продолжаем план улучшений backend Jessica Core.",
    history:[],
    message:[
      "JESSICA_REPLY_d1aafa0b-2726-4afc-84e0-d0d427a066f4",
      "PR #60 объединён, CI успешен. Неверный owner receipt не переводит "+
        "Proposal в FAILED. PR #59 исправил парсер Codex в GitHub. "+
        "Supabase здоров; Render ждёт workspace. "+
        "Тест Jessica не запускался. Следующий шаг — проверить backend в Render.",
      "JESSICA_END_d1aafa0b-2726-4afc-84e0-d0d427a066f4"
    ].join("\n")
  };
  const result=await interpret(payload,true);
  assert.equal(result.action,"reply");
  assert.equal(result.task,"");
  assert.equal(result.continueThread,false);
  assert.match(result.reply,/тест Jessica не выполнялся/);
  assert.match(result.reply,/Проверь статус через GitHub, Render и Supabase/);
}
