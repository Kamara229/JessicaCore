/**
 * Small deterministic safety checks for demonstrably broken generated source.
 * Static checks are NOT a compiler, and do not claim test execution.
 * Conservative routing: non-programming tasks and unrelated algorithms untouched.
 */
export function checkProgrammingAnswerSafety(task,answer) {
    const request=String(task||"");
    const text=String(answer||"");
    if (!text.trim()) return null;
    const factorial=/факториал|factorial/iu.test(request+" "+text);

    // Kotlin's signed Long represents factorial up to 20!, but 21! overflows.
    // A function using Long as result must reject n>20 or use BigInteger.
    const kotlin=[...text.matchAll(/\x60{3}kotlin\s*\n([\s\S]*?)\x60{3}/giu)].map(m=>m[1]);
    const kotlinSources=kotlin.length?kotlin:(/\bfun\s+factorial\s*\(/u.test(text)?[text]:[]);
    if (factorial && kotlinSources.some(source=>{
        if(!/\bfun\s+factorial\s*\(/u.test(source)) return false;
        if(!/\bLong\b/u.test(source)||/\bBigInteger\b/u.test(source)) return false;
        const hasRangeGuard =
            /(?:if\s*\(\s*n\s*>\s*20\s*\)|require\s*\(\s*n\s*<=\s*20\s*\)|check\s*\(\s*n\s*<=\s*20\s*\)|if\s*\(\s*n\s*!in\s*0\s*\.\.\s*20\s*\)|if\s*\(\s*n\s*>=\s*21\s*\))/u.test(source);
        return !hasRangeGuard;
    })){
        return {
            valid:false,shouldRetry:false,needsClarification:false,
            failureScope:"presentation",reason:
            "Функция factorial на Kotlin использует Long без ограничения n ≤ 20: для n ≥ 21 возникает переполнение. Используй BigInteger либо добавь проверку диапазона входных данных."
        };
    }

    // Detect one known syntax corruption: two quotes mistakenly used as the
    // opening Python function docstring, then unquoted narrative.
    const python=[...text.matchAll(/\x60{3}python\s*\n([\s\S]*?)\x60{3}/giu)].map(m=>m[1]);
    const pythonSources=python.length?python:(/^\s*def\s+\w+\s*\(/m.test(text)?[text]:[]);
    if (pythonSources.some(src=>
        /(?:^|\n)\s*def\s+\w+\s*\([^\n]*:\s*\n\s*""\s*\n\s*[А-Яа-яЁёA-Za-z][^\n]*\n/u.test(src)
    )){
        return {
            valid:false,shouldRetry:false,needsClarification:false,
            failureScope:"presentation",reason:
            "В Python используется неверный разделитель docstring: две кавычки вместо трёх; пояснение не является допустимой строкой Python."
        };
    }
    return null;
}
