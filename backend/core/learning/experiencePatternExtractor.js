/*
 * =========================================================
 * JESSICA EXPERIENCE PATTERN EXTRACTOR v1
 * =========================================================
 *
 * Автономное извлечение нового Experience Pattern
 * из успешного Execution Trace.
 *
 *
 * Flow:
 *
 * Unknown Successful Execution
 *        ↓
 * Learning Queue
 *        ↓
 * Learning Worker
 *        ↓
 * Experience Pattern Extractor
 *        ↓
 * AI Analysis
 *        ↓
 * Parse
 *        ↓
 * Structure Validation
 *        ↓
 * Generalization Validation
 *        ↓
 * Grounding Normalization
 *        ↓
 * Dynamic Experience Pattern
 *
 *
 * Используется только если:
 *
 * - Execution успешно завершён;
 * - Existing Experience не использовался;
 * - Known EXPERIENCE_PATTERN не найден.
 *
 *
 * Ответственность:
 *
 * - проанализировать успешный Execution;
 * - определить, содержит ли он
 *   переиспользуемую стратегию;
 * - обобщить стратегию;
 * - сформировать Dynamic Pattern;
 * - удалить task-specific значения;
 * - не позволить AI придумать
 *   инструменты, которые не использовались.
 *
 *
 * НЕ:
 *
 * - сохраняет Skill;
 * - пишет в Supabase;
 * - создаёт Learning Proposal;
 * - принимает AUTO_APPROVE;
 * - управляет версиями;
 * - изменяет Existing Experience;
 * - накапливает KEEP_CANDIDATE.
 *
 *
 * ВАЖНО:
 *
 * AI здесь создаёт только CANDIDATE PATTERN.
 *
 * Он НЕ получает права напрямую
 * записывать Experience в память Jessica.
 *
 * Результат всё равно проходит:
 *
 * Candidate Builder
 *      ↓
 * Proposal
 *      ↓
 * Reviewer
 *      ↓
 * Quality Gate
 *      ↓
 * Autonomy Policy
 *      ↓
 * Experience Storage
 *
 * =========================================================
 */


import OpenAI from "openai";


import {
    buildLearningSkillId
} from "../../experience/learning/learningSkillBuilder.js";





/*
 * =========================================================
 * CONFIG
 * =========================================================
 */


const MODEL =

    process.env.JESSICA_LEARNING_MODEL

    ||

    process.env.GROQ_MODEL

    ||

    "openai/gpt-oss-20b";



const MAX_TEXT_LENGTH =
    4000;



const MAX_WORKFLOW_STEPS =
    10;



const MAX_TRIGGER_PATTERNS =
    15;



const MAX_RULES =
    10;





/*
 * =========================================================
 * CLIENT
 * =========================================================
 */


function createClient()
{


    const apiKey =

        process.env.GROQ_API_KEY;



    if(
        !apiKey
    ){

        return null;

    }



    return new OpenAI({

        apiKey,


        baseURL:

            "https://api.groq.com/openai/v1"

    });

}





/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function isObject(
    value
)
{

    return (

        value &&

        typeof value === "object" &&

        !Array.isArray(value)

    );

}





function normalizeText(
    value,
    maxLength = MAX_TEXT_LENGTH
)
{

    return String(
        value || ""
    )
    .trim()
    .slice(
        0,
        maxLength
    );

}





function normalizeNumber(
    value
)
{

    const number =
        Number(value);



    return Number.isFinite(number)

        ? number

        : 0;

}





function normalizeUnit(
    value
)
{

    return Math.max(

        0,

        Math.min(

            1,

            normalizeNumber(
                value
            )

        )

    );

}





function normalizeStringArray(
    value,
    limit = 10
)
{

    if(
        !Array.isArray(value)
    ){

        return [];

    }



    const result = [];



    for(
        const item
        of value
    ){

        const normalized =

            normalizeText(
                item,
                500
            );



        if(
            !normalized
        ){

            continue;

        }



        if(
            result.includes(
                normalized
            )
        ){

            continue;

        }



        result.push(
            normalized
        );



        if(
            result.length >= limit
        ){

            break;

        }

    }



    return result;

}





/*
 * =========================================================
 * SAFE JSON STRINGIFY
 * =========================================================
 */


function safeJson(
    value
)
{

    try{


        return JSON.stringify(
            value
        );


    }catch(error){


        return String(
            value || ""
        );

    }

}





/*
 * =========================================================
 * ANSWER TEXT
 * =========================================================
 */


function extractAnswerText(
    trace
)
{

    const result =

        trace?.result;



    const candidates = [

        result?.answer?.text,

        result?.text,

        trace?.answer?.text,

        trace?.answer,

        trace?.resultText

    ];



    for(
        const candidate
        of candidates
    ){

        const text =

            normalizeText(
                candidate,
                3000
            );



        if(
            text
        ){

            return text;

        }

    }



    return "";

}





/*
 * =========================================================
 * USED TOOLS
 * =========================================================
 */


function extractUsedTools(
    trace
)
{

    const tools = [];



    const direct = [

        ...(Array.isArray(trace?.usedTools)
            ? trace.usedTools
            : []),

        ...(Array.isArray(trace?.tools)
            ? trace.tools
            : [])

    ];



    for(
        const item
        of direct
    ){

        const name =

            normalizeText(

                typeof item === "string"

                    ? item

                    : (
                        item?.name
                        ||
                        item?.tool
                        ||
                        item?.toolName
                    ),

                150
            );



        if(
            name &&
            !tools.includes(name)
        ){

            tools.push(
                name
            );

        }

    }



    const steps =

        Array.isArray(trace?.steps)

            ? trace.steps

            : [];



    for(
        const step
        of steps
    ){

        const name =

            normalizeText(

                step?.tool

                ||

                step?.toolName

                ||

                step?.metadata?.tool,

                150
            );



        if(
            name &&
            !tools.includes(name)
        ){

            tools.push(
                name
            );

        }

    }



    return tools.slice(
        0,
        20
    );

}





/*
 * =========================================================
 * TRACE STEPS
 * =========================================================
 */


function extractSteps(
    trace
)
{

    const steps =

        Array.isArray(trace?.steps)

            ? trace.steps

            : [];



    return steps

        .slice(
            -20
        )

        .map(

            step => ({

                stage:

                    normalizeText(
                        step?.stage ||
                        step?.name,
                        120
                    ),


                status:

                    normalizeText(
                        step?.status,
                        80
                    ),


                tool:

                    normalizeText(
                        step?.tool ||
                        step?.toolName ||
                        step?.metadata?.tool,
                        120
                    ),


                reason:

                    normalizeText(
                        step?.reason,
                        500
                    )

            })

        );

}





/*
 * =========================================================
 * FAILURES
 * =========================================================
 */


function extractFailures(
    trace
)
{

    const failures =

        Array.isArray(trace?.failures)

            ? trace.failures

            : [];



    return failures

        .slice(
            -10
        )

        .map(

            failure => ({

                stage:

                    normalizeText(
                        failure?.stage,
                        120
                    ),


                failureType:

                    normalizeText(
                        failure?.failureType,
                        150
                    ),


                category:

                    normalizeText(
                        failure?.category,
                        120
                    ),


                reason:

                    normalizeText(
                        failure?.reason,
                        700
                    )

            })

        );

}





/*
 * =========================================================
 * BUILD LEARNING EVIDENCE
 * =========================================================
 */


function buildLearningEvidence(
    trace,
    metrics = {}
)
{

    return {


        task:

            normalizeText(
                trace?.task,
                3000
            ),



        answer:

            extractAnswerText(
                trace
            ),



        result: {

            success:

                trace?.result?.success === true

                ||

                trace?.success === true,


            status:

                normalizeText(

                    trace?.result?.status

                    ||

                    trace?.status,

                    100
                ),


            terminalType:

                normalizeText(

                    trace
                        ?.result
                        ?.terminal
                        ?.type

                    ||

                    trace
                        ?.terminal
                        ?.type,

                    100
                )

        },



        validation: {

            valid:

                trace
                    ?.result
                    ?.validation
                    ?.valid === true

                ||

                trace
                    ?.validation
                    ?.valid === true,


            outcomeType:

                normalizeText(

                    trace
                        ?.result
                        ?.validation
                        ?.outcomeType

                    ||

                    trace
                        ?.validation
                        ?.outcomeType,

                    120
                ),


            reason:

                normalizeText(

                    trace
                        ?.result
                        ?.validation
                        ?.reason

                    ||

                    trace
                        ?.validation
                        ?.reason,

                    1000
                )

        },



        tools:

            extractUsedTools(
                trace
            ),



        steps:

            extractSteps(
                trace
            ),



        failures:

            extractFailures(
                trace
            ),



        replans:

            Array.isArray(
                trace?.replans
            )

                ? trace.replans.length

                : normalizeNumber(
                    trace?.statistics?.replans
                ),



        metrics: {

            occurrences:

                Math.max(
                    normalizeNumber(
                        metrics?.occurrences
                    ),
                    1
                ),


            successRate:

                normalizeUnit(
                    metrics?.successRate
                ),


            maturity:

                normalizeUnit(
                    metrics?.maturity
                )

        }

    };

}





/*
 * =========================================================
 * PROMPT
 * =========================================================
 */


function buildSystemPrompt()
{

    return `
Ты — внутренний модуль автономного обучения Jessica.

Твоя задача — определить, содержит ли УСПЕШНОЕ выполнение
переиспользуемую стратегию решения КЛАССА задач.

Ты НЕ создаёшь фактологическую память.
Ты создаёшь только обобщаемый процедурный Experience Pattern.

КРИТИЧЕСКИЕ ПРАВИЛА:

1. Не копируй конкретный ответ задачи в Skill.

2. Не сохраняй:
   - конкретные названия компаний;
   - имена людей;
   - конкретные URL и домены;
   - email;
   - телефоны;
   - UUID;
   - номера заказов;
   - уникальные идентификаторы;
   - конкретные значения, относящиеся только к одному примеру.

3. Workflow должен описывать общий способ решения,
   который можно применить к другим похожим задачам.

4. Trigger patterns должны описывать КЛАСС запросов,
   а не конкретную текущую задачу.

5. Validation rules должны объяснять,
   как проверить правильность результата.

6. Constraints должны описывать ограничения стратегии.

7. requiredTools можно выбирать ТОЛЬКО из инструментов,
   реально использованных в Execution Evidence.

8. failurePatterns и avoidPatterns можно указывать
   только если Execution Evidence реально содержит
   соответствующие ошибки, retry или replan.

9. Если Execution не содержит достаточно общего,
   переиспользуемого опыта — reusable=false.

10. Не придумывай знания, которых нет в Evidence.

Верни ТОЛЬКО JSON.

Формат:

{
  "reusable": true,
  "reason": "краткое объяснение",
  "pattern": {
    "name": "краткое название навыка",
    "category": "общая категория",
    "description": "что умеет этот Skill",
    "triggerPatterns": [
      "тип запроса 1",
      "тип запроса 2"
    ],
    "workflow": [
      "шаг 1",
      "шаг 2"
    ],
    "validationRules": [
      "правило проверки"
    ],
    "constraints": [
      "ограничение"
    ],
    "requiredTools": [
      "tool_name"
    ],
    "successfulPatterns": [
      "что оказалось полезной стратегией"
    ],
    "failurePatterns": [],
    "avoidPatterns": []
  }
}

Если опыта недостаточно:

{
  "reusable": false,
  "reason": "почему нельзя создать reusable Skill",
  "pattern": null
}
`.trim();

}





function buildUserPrompt(
    evidence
)
{

    return (

        "Execution Evidence:\n"

        +

        safeJson(
            evidence
        )

    );

}





/*
 * =========================================================
 * PARSE AI JSON
 * =========================================================
 */


function extractJsonText(
    rawText
)
{

    let text =

        normalizeText(
            rawText,
            20000
        );



    if(
        text.startsWith(
            "```"
        )
    ){

        text = text

            .replace(
                /^```(?:json)?\s*/i,
                ""
            )

            .replace(
                /\s*```$/,
                ""
            );

    }



    const firstBrace =

        text.indexOf(
            "{"
        );



    const lastBrace =

        text.lastIndexOf(
            "}"
        );



    if(
        firstBrace >= 0 &&
        lastBrace > firstBrace
    ){

        return text.slice(
            firstBrace,
            lastBrace + 1
        );

    }



    return text;

}





function parseExtractorResponse(
    rawText
)
{

    const jsonText =

        extractJsonText(
            rawText
        );



    try{


        return {

            success:
                true,


            data:

                JSON.parse(
                    jsonText
                )

        };


    }catch(error){


        return {

            success:
                false,


            data:
                null,


            error:

                error?.message

                ||

                "Invalid AI JSON"

        };

    }

}





/*
 * =========================================================
 * CONCRETE DATA DETECTION
 * =========================================================
 */


const CONCRETE_PATTERNS = [

    /https?:\/\/[^\s]+/i,

    /\bwww\.[^\s]+/i,

    /\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b/i,

    /\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b/i,

    /\b(?:\+?\d[\d\s\-()]{8,}\d)\b/

];





function containsConcreteData(
    value
)
{

    const text =

        normalizeText(
            value,
            10000
        );



    if(
        !text
    ){

        return false;

    }



    return CONCRETE_PATTERNS.some(

        pattern =>
            pattern.test(
                text
            )

    );

}





/*
 * =========================================================
 * VALIDATE GENERALIZATION
 * =========================================================
 */


function validateGeneralization(
    pattern
)
{

    const fields = [

        pattern?.name,

        pattern?.description,

        ...(Array.isArray(pattern?.triggerPatterns)
            ? pattern.triggerPatterns
            : []),

        ...(Array.isArray(pattern?.workflow)
            ? pattern.workflow
            : []),

        ...(Array.isArray(pattern?.validationRules)
            ? pattern.validationRules
            : []),

        ...(Array.isArray(pattern?.constraints)
            ? pattern.constraints
            : []),

        ...(Array.isArray(pattern?.successfulPatterns)
            ? pattern.successfulPatterns
            : []),

        ...(Array.isArray(pattern?.failurePatterns)
            ? pattern.failurePatterns
            : []),

        ...(Array.isArray(pattern?.avoidPatterns)
            ? pattern.avoidPatterns
            : [])

    ];



    const leaked =

        fields.filter(
            containsConcreteData
        );



    if(
        leaked.length > 0
    ){

        return {

            valid:
                false,


            reason:

                "Dynamic Pattern содержит конкретные данные единичного Execution"

        };

    }



    return {

        valid:
            true

    };

}





/*
 * =========================================================
 * NORMALIZE REQUIRED TOOLS
 * =========================================================
 *
 * AI не может добавить инструмент,
 * которого не было в Execution Evidence.
 *
 * =========================================================
 */


function normalizeRequiredTools(
    requestedTools,
    usedTools
)
{

    const requested =

        normalizeStringArray(
            requestedTools,
            20
        );



    const allowed =

        normalizeStringArray(
            usedTools,
            20
        );



    if(
        allowed.length === 0
    ){

        return [];

    }



    const allowedMap =

        new Map(

            allowed.map(

                item => [

                    item.toLowerCase(),

                    item

                ]

            )

        );



    const result = [];



    for(
        const item
        of requested
    ){

        const matched =

            allowedMap.get(
                item.toLowerCase()
            );



        if(
            matched &&
            !result.includes(matched)
        ){

            result.push(
                matched
            );

        }

    }



    return result;

}





/*
 * =========================================================
 * NORMALIZE PATTERN
 * =========================================================
 */


function normalizePattern(
    pattern,
    evidence
)
{

    if(
        !isObject(
            pattern
        )
    ){

        return null;

    }



    const name =

        normalizeText(
            pattern.name,
            160
        );



    if(
      
