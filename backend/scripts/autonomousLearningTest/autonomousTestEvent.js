/*
 * =========================================================
 * JESSICA AUTONOMOUS LEARNING TEST EVENT
 * =========================================================
 *
 * Execution Trace
 *      ↓
 * REAL Learning Trigger
 *      ↓
 * Learning Event
 *      ↓
 * diagnostic identity isolation
 *
 *
 * Workflow / Rules / Trigger Patterns
 * НЕ подменяются.
 *
 * =========================================================
 */


import {
    runLearningTrigger
} from "../../core/learning/learningTrigger.js";


import {
    AUTONOMOUS_TEST_SKILL_ID,
    AUTONOMOUS_TEST_SKILL_NAME
} from "./autonomousTestConstants.js";


function cloneValue(
    value
) {

    return JSON.parse(
        JSON.stringify(value)
    );

}


export function createAutonomousLearningTestEvent(
    trace
) {

    const triggerResult =

        runLearningTrigger(
            trace
        );


    if(
        !triggerResult?.triggered
    ){

        return {

            success:
                false,

            reason:

                triggerResult?.reason

                ||

                "Learning Trigger не запущен",

            trigger:
                triggerResult,

            event:
                null

        };

    }


    if(
        !triggerResult.learningEvent
    ){

        return {

            success:
                false,

            reason:
                "Learning Event отсутствует",

            trigger:
                triggerResult,

            event:
                null

        };

    }


    if(
        triggerResult.learningEvent.action ===
        "IGNORE"
    ){

        return {

            success:
                false,

            reason:
                "Analyzer вернул IGNORE",

            trigger:
                triggerResult,

            event:
                null

        };

    }


    /*
     * Для первого E2E используем только
     * детерминированный NEW_SKILL path.
     *
     * PATTERN_DISCOVERY проверим отдельно,
     * потому что он требует AI extraction.
     */


    if(
        triggerResult.learningEvent.action !==
        "NEW_SKILL"
    ){

        return {

            success:
                false,

            reason:

                `Ожидался NEW_SKILL, получен ${triggerResult.learningEvent.action}`,

            trigger:
                triggerResult,

            event:
                null

        };

    }


    const event =

        cloneValue(
            triggerResult.learningEvent
        );


    const candidate =

        event
            ?.payload
            ?.skillCandidate;


    if(
        !candidate
    ){

        return {

            success:
                false,

            reason:
                "Learning Event не содержит skillCandidate",

            trigger:
                triggerResult,

            event:
                null

        };

    }


    /*
     * =====================================================
     * DIAGNOSTIC ISOLATION
     * =====================================================
     *
     * Меняем только техническую identity,
     * чтобы тест никогда не обновил
     * production Skill.
     *
     * =====================================================
     */


    candidate.skillId =
        AUTONOMOUS_TEST_SKILL_ID;


    candidate.id =
        AUTONOMOUS_TEST_SKILL_ID;


    candidate.name =
        AUTONOMOUS_TEST_SKILL_NAME;


    candidate.category =
        "diagnostic";


    candidate.source =

        "autonomous-learning-e2e-test";


    /*
     * Если Event отдельно несёт skillId,
     * тоже синхронизируем.
     */


    event.skillId =
        AUTONOMOUS_TEST_SKILL_ID;


    event.payload.source =
        "autonomous-learning-e2e-test";


    event.reason =

        `${event.reason || ""} [diagnostic-e2e]`
            .trim();


    return {

        success:
            true,

        trigger:
            triggerResult,

        event

    };

}
