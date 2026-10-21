import test from "node:test";
import assert from "node:assert/strict";
import {checkProgrammingAnswerSafety} from "../core/validator/programmingAnswerSafety.js";
import {validateResult} from "../core/validator.js";

const fence=String.fromCharCode(96).repeat(3);
const kotlin=(body)=>fence+"kotlin\n"+body+"\n"+fence;

test("reject the Kotlin factorial implementation actually returned to the user",()=>{
    const code=kotlin([
        "fun factorial(n: Int): Long {",
        "    if (n < 0) throw IllegalArgumentException(\"Negative\")",
        "    var result = 1L",
        "    for (i in 2..n) result *= i",
        "    return result",
        "}"
    ].join("\n"));
    const rejected=checkProgrammingAnswerSafety(
        "Теперь добавь обработку неправильного ввода и перепиши функцию факториала на Kotlin",code);
    assert.equal(rejected?.failureScope,"presentation");
    assert.equal(rejected?.valid,false);
    assert.match(rejected?.reason,/переполнение/);
});

test("reject malformed Python double-quote function docstring",()=>{
    const broken="def factorial(n: int) -> int:\n    \"\"\n    Вычисляет факториал числа n.\n    \"\"\n    return 1";
    assert.match(checkProgrammingAnswerSafety("Напиши факториал на Python",broken)?.reason,/docstring/);
});

test("accept Kotlin factorial with explicit maximum and BigInteger variant",()=>{
    const safe=kotlin("fun factorial(n: Int): Long {\n  require(n <= 20)\n  var x = 1L\n  for(i in 2..n) x *= i\n  return x\n}");
    assert.equal(checkProgrammingAnswerSafety("Напиши factorial",safe),null);
    const big=kotlin("fun factorial(n: Int): BigInteger { return BigInteger.ONE }");
    assert.equal(checkProgrammingAnswerSafety("Напиши факториал",big),null);
});

test("ordinary code, non-programming requests and correct Python remain untouched",()=>{
    const correct="def factorial(n: int) -> int:\n    \"\"\"Return the factorial.\"\"\"\n    return 1";
    assert.equal(checkProgrammingAnswerSafety("Напиши factorial",correct),null);
    assert.equal(checkProgrammingAnswerSafety("Найди ресторан","Факториал — математическая функция."),null);
    assert.equal(checkProgrammingAnswerSafety("Напиши на Kotlin",kotlin("fun add(a:Int,b:Int):Long=a.toLong()+b")),null);
});

test("main Validator cannot mark dangerous factorial code as valid even if AI might",async()=>{
    const unsafe=kotlin("fun factorial(n: Int): Long {\n var result=1L\n for (i in 2..n) result *= i\n return result\n}");
    const result=await validateResult(
        "Перепиши факториал на Kotlin", {evidence:{mode:"none"}}, {success:true},
        {success:true,text:unsafe}
    );
    assert.equal(result.valid,false);
    assert.equal(result.failureScope,"presentation");
    assert.equal(result.validationMode,"programming-static-guard");
    assert.equal(result.shouldRetry,false);
});
