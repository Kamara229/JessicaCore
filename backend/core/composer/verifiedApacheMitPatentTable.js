const A="https://www.apache.org/licenses/LICENSE-2.0.txt";
const M="https://opensource.org/license/mit";
const whitespace=s=>String(s||"").replace(/\s+/gu," ").toLowerCase();
export function apacheMitPatentTable(task,apache,mit){
 if(!/(?:таблиц|таблич|table)/iu.test(String(task||"")))return null;
 const q1="patent license to make, have made, use";
 const q2="shall terminate as of the date such litigation is filed";
 const q3="Permission is hereby granted, free of charge";
 if(![q1,q2].every(q=>whitespace(apache).includes(q.toLowerCase()))||
    !whitespace(mit).includes(q3.toLowerCase()))return null;
 return [
  "| Критерий | Apache License 2.0 | MIT License |",
  "| --- | --- | --- |",
  "| Патентная лицензия | Прямо предоставлена в §3: «"+q1+"». | Прямой патентной лицензии нет. Разрешение: «"+q3+"». |",
  "| Прекращение | В §3 при подаче иска о патентном нарушении Work или Contribution, включая встречный: «"+q2+"». | Прямого правила о прекращении нет. |",
  "| Оговорка | Не любой патентный иск прекращает права. | Возможны подразумеваемые патентные права. |",
  "| Официальный текст | "+A+" | "+M+" |"
 ].join("\n");
}
