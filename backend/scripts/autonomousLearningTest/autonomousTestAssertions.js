/*
 * =========================================================
 * JESSICA AUTONOMOUS LEARNING TEST ASSERTIONS
 * =========================================================
 */


import {
    AUTONOMOUS_TEST_SKILL_ID
} from "./autonomousTestConstants.js";


function check(
    condition,
    name,
    details = null
) {

    return {

        name,

        passed:

            condition === true,

        details

    };

}


export function assertAutonomousLearningResult({

    daemonResult,

    queueItem,

    state

}) {

    const proposal =
        state?.proposal;


    const history =

        Array.isArray(
            state?.history
        )

            ? state.history

            : [];


    const activeSkill =
        state?.activeSkill;


    const tests = [];


    tests.push(

        check(

            daemonResult?.success === true,

            "daemon-cycle-success",

            daemonResult?.error || null

        )

    );


    tests.push(

        check(

            Boolean(queueItem?.id),

            "queue-item-created",

            queueItem?.id || null

        )

    );


    tests.push(

        check(

            Boolean(proposal),

            "proposal-created",

            proposal?.id || null

        )

    );


    tests.push(

        check(

            proposal?.status ===
            "APPROVED",

            "proposal-approved",

            proposal?.status || null

        )

    );


    tests.push(

        check(

            history.length === 1,

            "exactly-one-history-version",

            history.length

        )

    );


    const savedVersion =

        history[0];


    tests.push(

        check(

            savedVersion?.skillId ===
            AUTONOMOUS_TEST_SKILL_ID,

            "history-skill-id",

            savedVersion?.skillId || null

        )

    );


    tests.push(

        check(

            savedVersion
                ?.payload
                ?.metadata
                ?.proposalId
            ===
            proposal?.id,

            "history-proposal-id",

            {
                expected:
                    proposal?.id || null,

                actual:

                    savedVersion
                        ?.payload
                        ?.metadata
                        ?.proposalId

                    ||

                    null
            }

        )

    );


    tests.push(

        check(

            Boolean(activeSkill),

            "active-skill-loaded",

            activeSkill?.id || null

        )

    );


    tests.push(

        check(

            activeSkill?.id ===
            AUTONOMOUS_TEST_SKILL_ID,

            "active-skill-id",

            activeSkill?.id || null

        )

    );


    tests.push(

        check(

            activeSkill?.status ===
            "published",

            "active-skill-published",

            activeSkill?.status || null

        )

    );


    tests.push(

        check(

            activeSkill?.enabled === true,

            "active-skill-enabled",

            activeSkill?.enabled ?? null

        )

    );


    tests.push(

        check(

            Number(activeSkill?.version)
            ===
            Number(savedVersion?.version),

            "current-history-version-match",

            {
                current:
                    activeSkill?.version || null,

                history:
                    savedVersion?.version || null
            }

        )

    );


    const failed =

        tests.filter(

            item =>
                item.passed !== true

        );


    return {

        success:

            failed.length === 0,

        total:

            tests.length,

        passed:

            tests.length
            -
            failed.length,

        failed:

            failed.length,

        tests

    };

}
