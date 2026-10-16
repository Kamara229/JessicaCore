/*
 * =========================================================
 * JESSICA PATTERN REQUEST v2
 * =========================================================
 *
 * Координатор AI-запроса для автономного
 * Experience Pattern Extraction.
 *
 *
 * AI transport:
 *
 * backend/ai/patternExtractionClient.js
 *
 *
 * НЕ:
 *
 * - создаёт AI client;
 * - анализирует Execution Trace;
 * - парсит Pattern JSON;
 * - валидирует Pattern;
 * - сохраняет Experience.
 *
 * =========================================================
 */


import {
    isPatternExtractionAvailable,
    requestPatternExtractionAI
} from "../../../ai/patternExtractionClient.js";


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
 * SYSTEM PROMPT
 * =========================================================
 */


function buildSystemPrompt()
{

    return `
Ты — внутренний модуль автономного обучения Jessica.

Твоя задача — определить, содержит ли успешное выполнение
переиспользуемую стратегию решения КЛАССА задач.

Ты создаёшь только обобщаемый процедурный Experience Pattern.

Не создавай фактологическую память.

Правила:

1. Не копируй конкретный ответ задачи в Skill.

2. Не сохраняй конкретные:
- компании;
- имена людей;
- URL;
- домены;
- email;
- телефоны;
- UUID;
- номера заказов;
- уникальные значения единичного примера.

3. Workflow должен описывать общий способ решения.

4. Trigger patterns должны описывать класс запросов.

5. Validation rules должны объяснять,
как проверить правильность результата.

6. Constraints должны описывать ограничения стратегии.

7. requiredTools можно выбирать только
из реально использованных инструментов.

8. failurePatterns и avoidPatterns допускаются
только если Evidence содержит ошибки,
retry или replan.

9. Если общего переиспользуемого опыта недостаточно,
верни reusable=false.

10. Не придумывай знания,
которых нет в Execution Evidence.

Верни только JSON.

Формат:

{
  "reusable": true,
  "reason": "почему этот опыт переиспользуем",
  "pattern": {
    "name": "название навыка",
    "category": "категория",
    "description": "описание навыка",
    "triggerPatterns": [],
    "workflow": [],
    "validationRules": [],
    "constraints": [],
    "requiredTools": [],
    "successfulPatterns": [],
    "failurePatterns": [],
    "avoidPatterns": []
  }
}

Если опыта недостаточно:

{
  "reusable": false,
  "reason": "почему Skill создавать не следует",
  "pattern": null
}
`.trim();

}


/*
 * =========================================================
 * REQUEST
 * =========================================================
 */


export async function requestPatternExtraction(
    evidence
) {

    /*
     * =====================================================
     * AVAILABILITY
     * =====================================================
     */


    if(
        !isPatternExtractionAvailable()
    ){

        return {

            success:
                false,

            error:
                "GROQ_API_KEY отсутствует",

            rawText:
                ""

        };

    }


    /*
     * =====================================================
     * EVIDENCE
     * =====================================================
     */


    const evidenceText =

        safeJson(
            evidence
        );


    /*
     * =====================================================
     * AI REQUEST
     * =====================================================
     */


    try {


        const response =

            await requestPatternExtractionAI({

                systemPrompt:

                    buildSystemPrompt(),

                evidenceText

            });


        const rawText =

            response
                ?.choices
                ?.[0]
                ?.message
                ?.content

            ||

            "";


        if(
            !rawText
        ){

            return {

                success:
                    false,

                error:
                    "Pattern Extractor вернул пустой ответ",

                rawText:
                    ""

            };

        }


        return {

            success:
                true,

            rawText

        };


    }catch(error){


        console.error(

            "Jessica Pattern Extractor final error:",

            {

                status:
                    error?.status || 0,

                code:
                    error?.code || null,

                message:

                    error?.message

                    ||

                    String(error)

            }

        );


        return {

            success:
                false,

            error:

                error?.message

                ||

                "Pattern Extractor request failed",

            rawText:
                ""

        };

    }

}
