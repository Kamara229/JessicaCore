import test from "node:test";
import assert from "node:assert/strict";
import {correctNoticeFileClaim} from "../core/composer/comparisonConditionalNoticeMeaning.js";

const operative=[
  "4. Redistribution. You may reproduce and distribute copies of the Work or Derivative Works in Source or Object form, provided that You meet the following conditions:",
  "(c) You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work, excluding those notices that do not pertain to any part of the Derivative Works;",
  "(d) If the Work includes a NOTICE text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file, excluding those notices that do not pertain to any part of the Derivative Works."
].join(" ");
const notice="Если исходная работа содержит файл NOTICE, при распространении производной работы требуется включить читаемую копию относящихся к ней атрибуционных уведомлений из этого файла.";

test("repair the real regression: an unqualified all-notices claim and entire NOTICE content",()=>{
 const incorrect="При распространении необходимо сохранять все уведомления об авторском праве и, если присутствует файл NOTICE, включать его содержимое в распределения.";
 const result=correctNoticeFileClaim(incorrect,operative);
 assert.ok(result.includes(notice));
 assert.match(result,/в их исходной форме/);
 assert.match(result,/относящиеся к ним уведомления/);
 assert.doesNotMatch(result,/все уведомления|его содержимое/iu);
});

test("do not demand preservation or inclusion of the original NOTICE file",()=>{
 const incorrect="Требуется сохранять оригинальный файл **NOTICE** (если он присутствует) и включать его текст в распределения; также сохранять копирайт-уведомления.";
 const result=correctNoticeFileClaim(incorrect,operative);
 assert.ok(result.includes(notice));
 assert.doesNotMatch(result,/сохранять оригинальный файл|включать его текст/iu);
});

test("do not rewrite already limited notice wording or unrelated claims",()=>{
 const correct="Если исходная работа содержит файл NOTICE, при распространении производной работы требуется включить читаемую копию атрибуционных уведомлений из этого файла.";
 assert.equal(correctNoticeFileClaim(correct,operative),correct);
 const other="В исходном коде необходимо сохранить уведомление об авторском праве.";
 assert.equal(correctNoticeFileClaim(other,operative),other);
});

test("fail closed if the source lacks the conditional readable-copy obligation",()=>{
 const unsupported="При распространении необходимо сохранять все уведомления об авторском праве и включать содержимое файла NOTICE.";
 const absent=operative.replace(/must include a readable copy/iu,"may contain a copy");
 assert.equal(correctNoticeFileClaim(unsupported,absent),unsupported);
});

test("styled NOTICE content and inflected word формы are narrowed",()=>{
 const incorrect="При распространении необходимо воспроизводить содержание файла **NOTICE** (если он присутствует) и сохранять копирайт‑уведомления.\\u202f(«reproducing the content of the NOTICE file» — S1)";
 const corrected=correctNoticeFileClaim(incorrect,operative);
 assert.ok(corrected.includes(notice));
 assert.doesNotMatch(corrected,/воспроизводить содержание файла/iu);
});

test("do not rewrite supported conditional notices or denied duties",()=>{
 const qualified="Если исходная работа содержит файл NOTICE, при распространении производной работы требуется включить читаемую копию относящихся к ней атрибуционных уведомлений из этого файла.";
 assert.equal(correctNoticeFileClaim(qualified,operative),qualified);
 const denied="Ссылка на NOTICE файл приведена без обязательств по его сохранению.";
 assert.equal(correctNoticeFileClaim(denied,operative),denied);
});
