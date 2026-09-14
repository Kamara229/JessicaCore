/*
 * =========================================================
 * JESSICA EXPERIENCE SEARCH
 * =========================================================
 *
 * Отвечает только за поиск подходящего
 * накопленного опыта среди переданных Skills.
 *
 *
 * Вход:
 *
 * task
 * experiences[]
 *
 *
 * Выход:
 *
 * {
 *     found,
 *     experience,
 *     confidence,
 *     source
 * }
 *
 *
 * ВАЖНО:
 *
 * Этот модуль НЕ:
 *
 * - читает базу данных;
 * - сохраняет Experience;
 * - обучает Jessica;
 * - обращается к Planner;
 * - изменяет Skills.
 *
 *
 * Позже алгоритм поиска можно заменить:
 *
 * lexical
 *     ↓
 * semantic
 *     ↓
 * embeddings
 *     ↓
 * hybrid search
 *
 * не изменяя остальные части Jessica.
 *
 * =========================================================
 */


/*
 * =========================================================
 * SEARCH CONFIG
 * =========================================================
 */


/*
 * Минимальная уверенность,
 * при которой опыт считается подходящим.
 *
 * Пока используется простой поиск v0.1.
 *
 * В будущем порог можно перенести
 * в отдельную конфигурацию.
 */


const MIN_MATCH_CONFIDENCE =
    0.35;


/*
 * =========================================================
 * NORMALIZE TEXT
 * =========================================================
 */


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .toLowerCase()
        .replace(
            /[^a-zа-яё0-9\s_-]/gi,
            " "
        )
        .replace(
            /\s+/g,
            " "
        )
        .trim();

}


/*
 * =========================================================
 * TOKENIZE
 * =========================================================
 */


function tokenize(
    value
) {

    const normalized =
        normalizeText(
            value
        );


    if (!normalized) {

        return [];

    }


    return normalized
        .split(
            " "
        )
        .map(
            item =>
                item.trim()
        )
        .filter(
            item =>
                item.length >= 3
        );

}


/*
 * =========================================================
 * EXPERIENCE SEARCH TEXT
 * =========================================================
 *
 * Формирует только поисковое представление Skill.
 *
 * Основной алгоритм самого Skill
 * здесь НЕ используется.
 *
 * Для поиска достаточно метаданных:
 *
 * - name
 * - description
 * - taskTypes
 * - keywords
 * - tags
 *
 * =========================================================
 */


function buildExperienceSearchText(
    experience
) {

    if (
        !experience ||
        typeof experience !== "object"
    ) {

        return "";

    }


    const parts =
        [];


    if (experience.name) {

        parts.push(
            experience.name
        );

    }


    if (experience.description) {

        parts.push(
            experience.description
        );

    }


    const arrays = [

        experience.taskTypes,

        experience.keywords,

        experience.tags

    ];


    arrays.forEach(
        items => {

            if (
                Array.isArray(
                    items
                )
            ) {

                parts.push(
                    ...items
                );

            }

        }
    );


    return normalizeText(
        parts.join(
            " "
        )
    );

}


/*
 * =========================================================
 * CALCULATE MATCH
 * =========================================================
 *
 * Простая реализация v0.1.
 *
 * Сравнивает слова задачи
 * с поисковыми метаданными Skill.
 *
 *
 * ВАЖНО:
 *
 * Это временный поисковый алгоритм.
 *
 * Позже его заменим отдельным
 * Experience Matcher с семантическим поиском.
 *
 * =========================================================
 */


function calculateMatch(
    task,
    experience
) {

    const taskTokens =
        tokenize(
            task
        );


    if (
        taskTokens.length === 0
    ) {

        return 0;

    }


    const experienceText =
        buildExperienceSearchText(
            experience
        );


    if (!experienceText) {

        return 0;

    }


    let matched =
        0;


    taskTokens.forEach(
        token => {

            if (
                experienceText.includes(
                    token
                )
            ) {

                matched++;

            }

        }
    );


    return matched /
        taskTokens.length;

}


/*
 * =========================================================
 * SEARCH EXPERIENCE
 * =========================================================
 */


export function searchExperience(

    task,

    experiences = []

) {


    const cleanTask =
        String(
            task || ""
        ).trim();


    /*
     * =====================================================
     * INVALID INPUT
     * =====================================================
     */


    if (
        !cleanTask ||
        !Array.isArray(
            experiences
        ) ||
        experiences.length === 0
    ) {

        return {

            found:
                false,

            experience:
                null,

            confidence:
                0,

            source:
                "experience-search"

        };

    }


    /*
     * =====================================================
     * SEARCH
     * =====================================================
     */


    let bestExperience =
        null;


    let bestConfidence =
        0;


    for (
        const experience
        of experiences
    ) {


        /*
         * Можно временно отключить Skill,
         * не удаляя его из памяти Jessica.
         */


        if (
            experience?.enabled === false
        ) {

            continue;

        }


        const confidence =
            calculateMatch(
                cleanTask,
                experience
            );


        if (
            confidence >
            bestConfidence
        ) {

            bestExperience =
                experience;


            bestConfidence =
                confidence;

        }

    }


    /*
     * =====================================================
     * NO GOOD MATCH
     * =====================================================
     */


    if (
        !bestExperience ||
        bestConfidence <
            MIN_MATCH_CONFIDENCE
    ) {

        return {

            found:
                false,

            experience:
                null,

            confidence:
                bestConfidence,

            source:
                "experience-search"

        };

    }


    /*
     * =====================================================
     * MATCH FOUND
     * =====================================================
     */


    return {

        found:
            true,

        experience:
            bestExperience,

        confidence:
            bestConfidence,

        source:
            "experience-search"

    };

}
