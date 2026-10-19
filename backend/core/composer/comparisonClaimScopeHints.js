export function comparisonClaimScopeHints(reason){
 const text=String(reason||"");
 const hints=[];
 if(/пункт|раздел/i.test(text))
   hints.push("Убери номера пунктов без первичного подтверждения, опиши правовое действие.");
 if(/NOTICE|уведомлен|авторск/i.test(text))
   hints.push("Уточни обязанность по уведомлениям: исходная форма, читаемая копия и условность; не приравнивай сведения и файл.");
 if(/гаранти|ответственн/i.test(text))
   hints.push("Не объединяй отказ от гарантий с запретами без двух доказательств.");
 return hints.join("\n");
}
