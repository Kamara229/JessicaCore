/*
 * =========================================================
 * JESSICA AUTONOMOUS LEARNING TEST TRACE v2
 * =========================================================
 *
 * Второе независимое synthetic execution.
 *
 *
 * Цель:
 *
 * existing Candidate
 *        +
 * new independent evidence
 *        ↓
 * Candidate Merge
 *        ↓
 * occurrences = 2
 *        ↓
 * confidence ≈ 0.82
 *        ↓
 * AUTO_APPROVE
 *
 * =========================================================
 */


import {
    AUTONOMOUS_TEST_TASK,
    AUTONOMOUS_TEST_TRACE_ID,
    AUTONOMOUS_TEST_RESULT
} from "./autonomousTestConstants.js";


export function createAutonomousLearningTestTrace()
{

    const now =

        new Date()
            .toISOString();


    return {

        /*
         * =================================================
         * TRACE IDENTITY
         * =================================================
         */


        id:
            AUTONOMOUS_TEST_TRACE_ID,


        traceId:
            AUTONOMOUS_TEST_TRACE_ID,


        /*
         * =================================================
         * TASK
         * =================================================
         */


        task:
            AUTONOMOUS_TEST_TASK,


        /*
         * =================================================
         * EXECUTION STATE
         * =================================================
         */


        status:
            "COMPLETED",


        success:
            true,


        createdAt:
            now,


        completedAt:
            now,


        /*
         * =================================================
         * EXECUTION STATS
         * =================================================
         */


        stats: {

            completed:
                1,

            failed:
                0

        },


        /*
         * =================================================
         * EXPERIENCE USAGE
         * =================================================
         *
         * Нам нужен ещё один NEW_SKILL evidence
         * для существующего Candidate.
         *
         * Published Experience пока нет,
         * поэтому existing Skill не использовался.
         *
         * =================================================
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
         * =================================================
         * VERIFIED RESULT
         * =================================================
         */


        result: {

            success:
                true,

            status:
                "COMPLETED",

            answer:
                AUTONOMOUS_TEST_RESULT

        },


        validation: {

            valid:
                true,

            outcomeType:
                "verified_result",

            reason:
                "Diagnostic execution result is verified"

        },


        /*
         * =================================================
         * TOOLS
         * =================================================
         */


        usedTools: [

            "web_search",

            "web_fetch"

        ],


        tools: [

            "web_search",

            "web_fetch"

        ],


        /*
         * =================================================
         * EXECUTION STEPS
         * =================================================
         */


        steps: [

            {

                type:
                    "search",

                tool:
                    "web_search",

                success:
                    true,

                description:
                    "Поиск официального сайта Python"

            },

            {

                type:
                    "verification",

                tool:
                    "web_fetch",

                success:
                    true,

                description:
                    "Проверка официального домена python.org"

            }

        ],


        /*
         * =================================================
         * FAILURE SIGNALS
         * =================================================
         */


        failures:
            [],


        replans:
            []

    };

}
