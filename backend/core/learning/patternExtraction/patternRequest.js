/*
 * =========================================================
 * JESSICA PATTERN REQUEST v1
 * =========================================================
 *
 * AI запрос для автономного
 * Experience Pattern Extraction.
 *
 *
 * НЕ:
 *
 * - анализирует Execution Trace;
 * - парсит JSON;
 * - валидирует Pattern;
 * - сохраняет Experience.
 *
 * =========================================================
 */


import OpenAI from "openai";





const MODEL =

    process.env.JESSICA_LEARNING_MODEL

    ||

    process.env.GROQ_MODEL

    ||

    "openai/gpt-oss-20b";





/*
 * =========================================================
 * CLIENT
 * =========================================================
 */


function createClient()
{

    if(
        !process.env.GROQ_API_KEY
    ){

        return null;

    }



    return new OpenAI({

        apiKey:

            process.env.GROQ_API_KEY,


        baseURL:

            "https://api.groq.com/openai/v1"

    });

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

    const client =

        createClient();



    if(
        !client
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



    try {


        const response =

            await client
                .chat
                .completions
                .create({

                    model:
                        MODEL,


                    temperature:
                        0.1,


                    messages: [

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

                                "Execution Evidence:\n"

                                +

                                safeJson(
                                    evidence
                                )

                        }

                    ]

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
