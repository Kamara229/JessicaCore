/*
 * =========================================================
 * JESSICA EXPERIENCE TEXT v0.4
 * =========================================================
 *
 * Текстовый слой Experience Search.
 *
 *
 * Отвечает за:
 *
 * - нормализацию текста;
 * - токенизацию;
 * - RU / EN semantic concepts;
 * - нормализацию массивов строк;
 * - удаление дублей.
 *
 *
 * НЕ отвечает за:
 *
 * - scoring;
 * - confidence;
 * - выбор Skill;
 * - phrase matching;
 * - Storage;
 * - AI.
 *
 * =========================================================
 */


/*
 * =========================================================
 * NORMALIZE TEXT
 * =========================================================
 */


export function normalizeExperienceText(
    value
) {

    return String(
        value || ""
    )
        .toLowerCase()

        /*
         * Для Experience Search дефисы
         * и подчёркивания считаем разделителями.
         */

        .replace(
            /[_-]+/g,
            " "
        )

        .replace(
            /[^a-zа-яё0-9\s]/gi,
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


export function tokenizeExperienceText(
    value
) {

    const text =
        normalizeExperienceText(
            value
        );


    if (!text) {

        return [];

    }


    return text
        .split(" ")
        .filter(
            token =>
                token.length >= 2
        );

}


/*
 * =========================================================
 * CANONICAL CONCEPT
 * =========================================================
 *
 * Приводит распространённые языковые формы
 * к стабильным semantic concepts.
 *
 *
 * Это НЕ полноценный NLP.
 *
 * Здесь должны находиться только
 * универсальные смысловые соответствия.
 *
 * Не добавляем сюда правила
 * под один конкретный пользовательский запрос.
 *
 * =========================================================
 */


export function canonicalizeExperienceToken(
    token
) {

    const value =
        normalizeExperienceText(
            token
        );


    if (!value) {

        return "";

    }


    /*
     * =====================================================
     * CURRENT
     * =====================================================
     *
     * current
     * now
     * текущий
     * текущее
     * текущего
     * сейчас
     *
     * → current
     */


    if (
        value === "current" ||
        value === "currently" ||
        value === "now" ||
        value === "сейчас" ||
        value.startsWith(
            "текущ"
        )
    ) {

        return "current";

    }


    /*
     * =====================================================
     * TIME
     * =====================================================
     *
     * time
     * hour
     * время
     * времени
     * час
     * часов
     *
     * → time
     */


    if (
        value === "time" ||
        value === "hour" ||
        value === "hours" ||
        value.startsWith(
            "врем"
        ) ||
        value.startsWith(
            "час"
        )
    ) {

        return "time";

    }


    /*
     * =====================================================
     * OFFICIAL
     * =====================================================
     */


    if (
        value === "official" ||
        value === "officially" ||
        value.startsWith(
            "официал"
        )
    ) {

        return "official";

    }


    /*
     * =====================================================
     * WEBSITE
     * =====================================================
     */


    if (
        value === "website" ||
        value === "site" ||
        value === "web" ||
        value.startsWith(
            "сайт"
        )
    ) {

        return "website";

    }


    /*
     * =====================================================
     * SEARCH / FIND
     * =====================================================
     */


    if (
        value === "search" ||
        value === "find" ||
        value === "finding" ||
        value.startsWith(
            "поиск"
        ) ||
        value.startsWith(
            "ищ"
        ) ||
        value.startsWith(
            "найд"
        ) ||
        value.startsWith(
            "найт"
        )
    ) {

        return "search";

    }


    /*
     * =====================================================
     * VERIFY
     * =====================================================
     */


    if (
        value === "verify" ||
        value === "verified" ||
        value === "verification" ||
        value.startsWith(
            "провер"
        )
    ) {

        return "verify";

    }


    /*
     * =====================================================
     * DOMAIN
     * =====================================================
     */


    if (
        value === "domain" ||
        value.startsWith(
            "домен"
        )
    ) {

        return "domain";

    }


    /*
     * =====================================================
     * URL / LINK
     * =====================================================
     */


    if (
        value === "url" ||
        value === "link" ||
        value.startsWith(
            "ссыл"
        )
    ) {

        return "url";

    }


    /*
     * =====================================================
     * PROJECT
     * =====================================================
     */


    if (
        value === "project" ||
        value.startsWith(
            "проект"
        )
    ) {

        return "project";

    }


    /*
     * =====================================================
     * FALLBACK
     * =====================================================
     *
     * Неизвестные слова не выбрасываем.
     *
     * Они всё ещё могут быть полезны
     * для будущих Skills:
     *
     * blender
     * android
     * invoice
     * seo
     * etc.
     *
     * =====================================================
     */


    return value;

}


/*
 * =========================================================
 * CANONICAL TOKENS
 * =========================================================
 */


export function canonicalizeExperienceTokens(
    value
) {

    return tokenizeExperienceText(
        value
    )
        .map(
            canonicalizeExperienceToken
        )
        .filter(Boolean);

}


/*
 * =========================================================
 * UNIQUE VALUES
 * =========================================================
 */


export function uniqueExperienceValues(
    values
) {

    if (
        !Array.isArray(
            values
        )
    ) {

        return [];

    }


    return [
        ...new Set(
            values.filter(Boolean)
        )
    ];

}


/*
 * =========================================================
 * NORMALIZE STRING ARRAY
 * =========================================================
 */


export function normalizeExperienceStringArray(
    value
) {

    if (
        !Array.isArray(
            value
        )
    ) {

        return [];

    }


    return value
        .map(
            item =>
                String(
                    item || ""
                ).trim()
        )
        .filter(Boolean);

}
