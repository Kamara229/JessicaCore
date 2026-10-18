/*
 * =========================================================
 * JESSICA LEARNING EVENT LOGGER
 * =========================================================
 */


export function logLearningEvents(
    events
) {

    const source =

        Array.isArray(
            events
        )

            ? events

            : [];


    for(
        const event
        of source
    ){

        console.log(

            "Jessica Learning Event:",

            JSON.stringify({

                action:

                    event?.action

                    ||

                    "IGNORE",

                skillId:

                    event?.skillId

                    ||

                    null,

                confidence:

                    Number(
                        event?.confidence || 0
                    ),

                reusable:

                    event?.reusable === true,

                hasCandidate:

                    Boolean(
                        event
                            ?.payload
                            ?.skillCandidate
                    ),

                existingSkills:

                    Array.isArray(
                        event
                            ?.payload
                            ?.skills
                    )

                        ? event
                            .payload
                            .skills
                            .length

                        : 0

            })

        );

    }

}
