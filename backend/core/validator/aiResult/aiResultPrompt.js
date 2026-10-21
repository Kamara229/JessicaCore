import {buildProgrammingFoundation} from "../../foundations/programmingPolicy.js";

/*
 * =========================================================
 * JESSICA AI RESULT PROMPT v2
 * =========================================================
 *
 * Формирует компактный prompt для AI Result Validator.
 *
 * Дополнительно требует failureScope, чтобы отличать:
 *
 * presentation
 * execution
 * evidence
 * clarification
 *
 * =========================================================
 */

import {
    buildCompactValidatorInput
} from "../validatorInput.js";


const MAX_TASK_LENGTH = 3000;
const MAX_PLAN_LENGTH = 5000;
const MAX_RUN_RESULT_LENGTH = 12000;
const MAX_ANSWER_LENGTH = 5000;


function limitText(
    value,
    maxLength
) {

    const text =
        String(
            value || ""
        ).trim();


    if(
        text.length <= maxLength
    ){

        return text;

    }


    return (
        text.slice(
            0,
            maxLength
        )
        +
        "…"
    );

}


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


function limitJson(
    value,
    maxLength
) {

    const json =
        safeJson(
            value
        );


    if(
        json.length <= maxLength
    ){

        return json;

    }


    return (
        json.slice(
            0,
            maxLength
        )
        +
        "\n...[truncated]"
    );

}


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
                    input.task.length
                    +
                    input.plan.length
                    +
                    input.runResult.length
                    +
                    input.answer.length
                )

        })
    );

}


function buildSystemPrompt(task = "")
{

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
        "- решает ли итоговый ответ исходную задачу;",
        "- соответствует ли итоговый ответ данным выполнения;",
        "- корректны ли сами данные выполнения;",
        "- достаточно ли evidence;",
        "- требуется ли уточнение пользователя.",

        "",

        "Если valid=false, обязательно определи failureScope.",

        "",

        "Допустимые failureScope:",

        "presentation:",
        "- данные выполнения достаточны и корректны;",
        "- проблема только в итоговом тексте;",
        "- ответ пропустил, исказил или отрицает уже полученный результат;",
        "- для исправления НЕ нужно повторять Planner, Tools или Execution.",

        "",

        "execution:",
        "- ошибка находится в самом результате выполнения, вычислении или логике решения;",
        "- одного переписывания итогового текста недостаточно.",

        "",

        "evidence:",
        "- результат или утверждение не имеет требуемого подтверждения;",
        "- отсутствует, недостаточен или противоречив evidence.",

        "",

        "clarification:",
        "- исходная задача действительно требует дополнительного ввода пользователя.",

        "",

        "КРИТИЧЕСКО:",
        "- если вычисление/результат уже корректно присутствует в Plan или Run Result, а итоговый ответ его отрицает или не сообщает, failureScope=presentation;",
        "- не назначай execution только потому, что итоговый текст неправильный;",
        "- если failureScope=presentation, shouldRetry=false: повтор Execution не нужен;",
        "- если failureScope=clarification, needsClarification=true;",
        "- если valid=true, failureScope должен быть null.",

        "",

        "Также определи outcomeType.",

        "Допустимы только:",
        "result",
        "no_verified_result",

        "",

        "result используй, если искомый результат найден, установлен или подтверждён.",

        "no_verified_result используй, если корректный итоговый ответ сообщает, что результат не удалось установить по имеющимся данным.",

        "",

        "Важно:",
        "- no_verified_result не означает ошибку;",
        "- valid может быть true одновременно с outcomeType=no_verified_result;",
        "- отсутствие подтверждения нельзя превращать в выдуманный положительный результат;",
        "- не отклоняй короткий корректный ответ.",

        "",

        "Верни только JSON следующей структуры:",

        JSON.stringify({

            valid:
                false,

            shouldRetry:
                false,

            needsClarification:
                false,

            failureScope:
                "presentation",

            outcomeType:
                "result",

            reason:
                "краткая причина"

        })

    ]
        .join(
            "\n"
        ) + "\n" + buildProgrammingFoundation(task,{stage:"validator"});

}


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
                buildSystemPrompt(task)

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
