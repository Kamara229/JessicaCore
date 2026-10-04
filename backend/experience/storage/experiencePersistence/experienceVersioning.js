/*
 * =========================================================
 * JESSICA EXPERIENCE VERSIONING v1
 * =========================================================
 *
 * Только работа с version state.
 *
 * НЕ:
 *
 * - читает БД;
 * - сохраняет Skill.
 *
 * =========================================================
 */


export function resolveExperienceVersionState(
    history
) {

    const items =

        Array.isArray(history)

            ? history

            : [];


    const versions =

        items

            .map(
                item =>
                    Number(
                        item?.version
                    )
            )

            .filter(
                value =>
                    Number.isInteger(value)
                    &&
                    value > 0
            );


    if(
        versions.length === 0
    ){

        return {

            exists:
                false,

            latestVersion:
                null,

            nextVersion:
                1,

            previousVersion:
                null

        };

    }


    const latestVersion =

        Math.max(
            ...versions
        );


    return {

        exists:
            true,

        latestVersion,

        nextVersion:
            latestVersion + 1,

        previousVersion:
            latestVersion

    };

}
