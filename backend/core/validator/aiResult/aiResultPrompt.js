/*
 * =========================================================
 * JESSICA AI RESULT PROMPT
 * =========================================================
 *
 * Формирует компактный prompt
 * для AI Result Validator.
 *
 *
 * Отвечает за:
 *
 * - подготовку входных данных;
 * - ограничение размеров;
 * - использование compact TaskRunner result;
 * - system prompt;
 * - user prompt;
 * - диагностику размера input.
 *
 *
 * НЕ отвечает за:
 *
 * - вызов AI;
 * - retry;
 * - parsing ответа;
 * - semantic normalization результата.
 *
 * =========================================================
 */


import {
    buildCompactValidatorInput
} from "../validatorInput.js";


/*
 * =========================================================
 * INPUT LIMITS
 * =========================================================
 */


const MAX_TASK_LENGTH =
    3000;


const MAX_PLAN_LENGTH =
    5000;


const MAX_RUN_RESULT_LENGTH =
    12000;


const MAX_ANSWER_LENGTH =
    5000;


/*
 * =========================================================
 * LIMIT TEXT
 * =========================================================
 */


function limitText(
    value,
    maxLength
) {

    const text =
        String(
            value || ""
        ).trim();


    if (
        text.length <= maxLength
    ) {

        return text;

    }


    return (
        text.slice(
            0,
            maxLength
        ) +
        "…"
    );

}


/*
 * =========================================================
 * SAFE JSON
 * =========================================================
 */


function safeJson(
    value
) {

    try {

        return JSON.stringify(
            value,
            null,
            2
        );

    } catch {

        return "{}";

    }

}


/*
 * =========================================================
 * LIMIT JSON
 * =========================================================
 */


function limitJson(
    value,
    maxLength
) {

    const json =
        safeJson(
            value
        );


    if (
        json.length <= maxLength
    ) {

        return json;

    }


    return (
        json.slice(
            0,
            maxLength
        ) +
        "\n...[truncated]"
    );

}


/*
 * =========================================================
 * PREPARE INPUT
 * =========================================================
 */


function prepareInput(
    task,
    plan,
    taskRunResult,
    answerResult
) {

    const compactRunResult =
        buildCompactValidatorInput(
            taskRunResult
        );


    return {

        task:
            limitText(
                task,
                MAX_TASK_LENGTH
            ),


        plan:
            limitJson(
                plan,
                MAX_PLAN_LENGTH
            ),


        runResult:
            limitJson(
                compactRunResult,
                MAX_RUN_RESULT_LENGTH
            ),


        answer:
            limitText(
                answerResult?.text,
                MAX_ANSWER_LENGTH
            )

    };

}


/*
 * =========================================================
 * LOG INPUT SIZE
 * =========================================================
 *
 * Это количество символов, НЕ tokens.
 *
 * Лог нужен для диагностики случаев,
 * подобных Groq 413.
 *
 * =========================================================
 */


function logInputSize(
    input
) {

    console.log(
        "Jessica Validator compact input:",
        JSON.stringify({

            taskChars:
                input.task.length,

            planChars:
                input.plan.length,

            runResultChars:
                input.runResult.length,

            answerChars:
                input.answer.length,

            totalChars:
                (
                    input.task.length +
                    input.plan.length +
                    input.runResult.length +
                    input.answer.length
                )

        })
    );

}


/*
 * =========================================================
 * SYSTEM PROMPT
 * =========================================================
 */


function buildSystemPrompt() {

    return [

        "Ты AI Validator системы Jessica Core.",

        "",

        "Ты проверяешь только уже полученный результат.",

        "",

        "Ты НЕ:",

        "- отвечаешь пользователю;",
        "- выполняешь инструменты;",
        "- ищешь новую информацию;",
        "- изменяешь план.",

        "",

        "Проверь:",

        "- решает ли ответ исходную задачу;",
        "- соответствует ли ответ фактическим данным выполнения;",
        "- есть ли критические ошибки;",
        "- нужна ли повторная попытка;",
        "- требуется ли уточнение пользователя.",

        "",

        "Также определи outcomeType.",

        "",

        "Допустимы только два outcomeType:",

        "",

        "1. result",

        "Используй result, если в данных выполнения есть достаточные основания считать, что искомый пользователем результат найден, установлен или подтверждён.",

        "",

        "2. no_verified_result",

        "Используй no_verified_result, если итоговый ответ корректно сообщает, что искомый результат не удалось найти, подтвердить или установить по имеющимся данным.",

        "",

        "Важно:",

        "- no_verified_result НЕ означает, что ответ неправильный;",
        "- valid может быть true одновременно с outcomeType=no_verified_result;",
        "- отсутствие подтверждения нельзя превращать в выдуманный положительный результат;",
        "- наличие поисковой выдачи само по себе не означает подтверждённый результат;",
        "- сторонний или неподтверждённый источник не считается подтверждением;",
        "- если искомый результат подтверждён имеющимися данными, используй result.",

        "",

        "Не отклоняй короткий корректный ответ.",

        "",

        "Верни только JSON следующей структуры:",

        JSON.stringify({

            valid:
                true,

            shouldRetry:
                false,

            needsClarification:
                false,

            outcomeType:
                "result",

            reason:
                "краткая причина"

        })

    ]
        .join(
            "\n"
        );

}


/*
 * =========================================================
 * USER PROMPT
 * =========================================================
 */


function buildUserPrompt(
    input
) {

    return [

        "ИСХОДНАЯ ЗАДАЧА:",

        input.task,


        "",


        "ПЛАН:",

        input.plan,


        "",


        "КОМПАКТНЫЙ РЕЗУЛЬТАТ ВЫПОЛНЕНИЯ:",

        input.runResult,


        "",


        "ИТОГОВЫЙ ОТВЕТ:",

        input.answer

    ]
        .join(
            "\n"
        );

}


/*
 * =========================================================
 * PUBLIC
 * =========================================================
 */


export function buildAIResultValidatorMessages(

    task,

    plan,

    taskRunResult,

    answerResult

) {

    const input =
        prepareInput(

            task,

            plan,

            taskRunResult,

            answerResult

        );


    logInputSize(
        input
    );


    return [

        {

            role:
                "system",

            content:
                buildSystemPrompt()

        },


        {

            role:
                "user",

            content:
                buildUserPrompt(
                    input
                )

        }

    ];

}
