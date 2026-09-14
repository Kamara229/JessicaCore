import {
    getToolCount,
    listTools
} from "../../tools/toolRegistry.js";

import {
    testSupabaseConnection
} from "../../storage/supabaseHealth.js";


/*
 * =========================================================
 * HEALTH ROUTE
 * =========================================================
 *
 * Проверяет состояние основных компонентов Jessica.
 *
 * Значения секретных ключей
 * никогда не возвращаются.
 *
 * =========================================================
 */


export function registerHealthRoute(
    app
) {

    app.get(
        "/api/health",

        async (
            req,
            res
        ) => {


            /*
             * =================================================
             * TOOLS
             * =================================================
             */


            const tools =
                listTools();


            /*
             * =================================================
             * DATABASE
             * =================================================
             */


            const supabase =
                await testSupabaseConnection();


            /*
             * =================================================
             * RESPONSE
             * =================================================
             */


            return res.json({

                success:
                    true,

                service:
                    "Jessica Core",

                version:
                    "1.0",

                architecture:
                    "planner-runner-tools-composer-validator",


                configured: {

                    appToken:
                        Boolean(
                            process.env.JESSICA_APP_TOKEN
                        ),

                    groq:
                        Boolean(
                            process.env.GROQ_API_KEY
                        ),

                    tinyFish:
                        Boolean(
                            process.env.TINYFISH_API_KEY
                        ),

                    openAI:
                        Boolean(
                            process.env.OPENAI_API_KEY
                        ),

                    supabase:
                        Boolean(
                            process.env.SUPABASE_URL &&
                            process.env.SUPABASE_SECRET_KEY
                        )

                },


                database: {

                    supabase

                },


                tools: {

                    count:
                        getToolCount(),

                    registered:
                        tools.map(
                            tool =>
                                tool.name
                        )

                }

            });

        }
    );

}
