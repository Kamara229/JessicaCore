/*
 * =========================================================
 * JESSICA AUTONOMOUS LEARNING TEST TRACE
 * =========================================================
 *
 * Создаёт synthetic successful Execution Trace.
 *
 * Он должен выглядеть как обычный успешный
 * runtime execution для Learning Analyzer.
 *
 * =========================================================
 */


import {
    AUTONOMOUS_TEST_TASK,
    AUTONOMOUS_TEST_TRACE_ID
} from "./autonomousTestConstants.js";


export function createAutonomousLearningTestTrace()
{

    const now =
        new Date()
            .toISOString();


    return {

        id:
            AUTONOMOUS_TEST_TRACE_ID,


        traceId:
            AUTONOMOUS_TEST_TRACE_ID,


        task:
            AUTONOMOUS_TEST_TASK,


        status:
            "COMPLETED",


        success:
            true,


        createdAt:
            now,


        completedAt:
            now,


        /*
         * Learning Analyzer сейчас
         * использует completed > 0
         * как один из сигналов успешности.
         */


        stats: {

            completed:
                1,

            failed:
                0

        },


        /*
         * Existing Experience намеренно
         * считаем не использованным.
         *
         * Нам нужен NEW_SKILL /
         * pattern path, а не improvement.
         */


        experienceUsage: {

            used:
                false,

            found:
                false,

            source:
                "diagnostic",

            confidence:
                0,

            skills:
                []

        },


        /*
         * Успешный проверенный результат.
         */


        result: {

            success:
                true,

            status:
                "COMPLETED",

            answer:
                "Официальный сайт Blender: https://www.blender.org/"

        },


        validation: {

            valid:
                true,

            outcomeType:
                "verified_result",

            reason:
                "Diagnostic execution result is verified"

        },


        usedTools: [

            "web_search",

            "web_fetch"

        ],


        tools: [

            "web_search",

            "web_fetch"

        ],


        steps: [

            {
                type:
                    "search",

                tool:
                    "web_search",

                success:
                    true,

                description:
                    "Поиск официального источника"
            },

            {
                type:
                    "verification",

                tool:
                    "web_fetch",

                success:
                    true,

                description:
                    "Проверка официального сайта"
            }

        ],


        failures:
            [],


        replans:
            []

    };

}
