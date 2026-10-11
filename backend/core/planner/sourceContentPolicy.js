/*
 * =========================================================
 * JESSICA SOURCE CONTENT POLICY v1
 * =========================================================
 *
 * Детерминированная policy для Planner.
 *
 *
 * Гарантирует:
 *
 * source_content
 *      ⇄
 * web_fetch
 *
 *
 * Также определяет случаи,
 * когда пользователь ЯВНО требует
 * открыть / прочитать / проверить
 * содержимое конкретного источника.
 *
 *
 * Важно:
 *
 * "найди на официальном сайте"
 *
 * само по себе НЕ означает,
 * что web_fetch обязателен.
 *
 *
 * Но:
 *
 * "открой страницу"
 * "проверь содержимое страницы"
 * "прочитай документ"
 *
 * требует source_content + web_fetch.
 *
 * =========================================================
 */


/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function normalizeText(
    value
) {

    return String(
        value || ""
    )
        .trim()
        .toLowerCase();

}


/*
 * =========================================================
 * EXPLICIT SOURCE CONTENT REQUIREMENT
 * =========================================================
 */


export function taskRequiresSourceContent(
    task
) {

    const text =

        normalizeText(
            task
        );


    if(
        !text
    ){

        return false;

    }


    const patterns = [

        /*
         * =================================================
         * RUSSIAN
         * =================================================
         */


        /\bоткрой\b/u,

        /\bоткрыть\b/u,

        /\bоткройте\b/u,

        /\bперейди\b/u,

        /\bперейти\b/u,

        /\bпрочитай\b/u,

        /\bпрочитать\b/u,

        /\bпрочти\b/u,


        /*
         * Проверка именно содержимого.
         */


        /проверь.{0,100}(содержим|текст|страниц|документ)/u,

        /проверить.{0,100}(содержим|текст|страниц|документ)/u,

        /(по|на основании|исходя из).{0,40}(содержим|текст).{0,40}(страниц|документ|источник)/u,

        /информац.{0,40}по содержим/u,

        /сверь.{0,80}(страниц|документ|источник)/u,

        /сверить.{0,80}(страниц|документ|источник)/u,


        /*
         * =================================================
         * ENGLISH
         * =================================================
         */


        /\bopen\s+(?:the\s+)?(?:official\s+)?(?:page|site|document|source)\b/i,

        /\bread\s+(?:the\s+)?(?:page|document|source)\b/i,

        /\b(?:check|verify).{0,80}(?:page|document|source).{0,80}(?:content|text)\b/i,

        /\b(?:according to|based on).{0,60}(?:page|document|source).{0,40}(?:content|text)\b/i

    ];


    return patterns.some(
        pattern =>
            pattern.test(
                text
            )
    );

}


/*
 * =========================================================
 * TOOL CHECK
 * =========================================================
 */


export function planUsesTool(
    plan,
    toolName
) {

    const name =

        String(
            toolName || ""
        )
            .trim();


    if(
        !name
        ||
        !Array.isArray(
            plan?.steps
        )
    ){

        return false;

    }


    return plan.steps.some(
        step =>
            String(
                step?.tool || ""
            )
                .trim()
                === name
    );

}


/*
 * =========================================================
 * POLICY VALIDATION
 * =========================================================
 */


export function validateSourceContentPolicy({

    plan,

    task = ""

} = {}) {

    const evidenceMode =

        String(
            plan?.evidence?.mode || ""
        )
            .trim()
            .toLowerCase();


    const usesFetch =

        planUsesTool(
            plan,
            "web_fetch"
        );


    const explicitSourceRequirement =

        taskRequiresSourceContent(
            task
        );


    /*
     * =====================================================
     * SOURCE_CONTENT REQUIRES WEB_FETCH
     * =====================================================
     */


    if(
        evidenceMode === "source_content"
        &&
        !usesFetch
    ){

        return {

            success:
                false,

            text:

                "evidence.mode=source_content требует шага web_fetch. "
                +
                "Если URL неизвестен, сначала используй web_search, "
                +
                "затем web_fetch с зависимостью $from."

        };

    }


    /*
     * =====================================================
     * WEB_FETCH MEANS SOURCE_CONTENT
     * =====================================================
     *
     * Если Planner реально загружает страницу,
     * её содержимое должно участвовать
     * в специализированной validation chain.
     * =====================================================
     */


    if(
        usesFetch
        &&
        evidenceMode !== "source_content"
    ){

        return {

            success:
                false,

            text:

                "План использует web_fetch, поэтому evidence.mode должен быть source_content."

        };

    }


    /*
     * =====================================================
     * EXPLICIT USER REQUIREMENT
     * =====================================================
     */


    if(
        explicitSourceRequirement
        &&
        evidenceMode !== "source_content"
    ){

        return {

            success:
                false,

            text:

                "Пользователь явно требует открыть, прочитать или проверить содержимое источника. "
                +
                "Используй evidence.mode=source_content и web_fetch."

        };

    }


    if(
        explicitSourceRequirement
        &&
        !usesFetch
    ){

        return {

            success:
                false,

            text:

                "Задача требует проверки содержимого источника, но web_fetch отсутствует. "
                +
                "Если адрес страницы заранее неизвестен, сначала найди URL через web_search, "
                +
                "после чего открой его через web_fetch."

        };

    }


    return {

        success:
            true,

        requiresSourceContent:
            explicitSourceRequirement,

        usesFetch,

        evidenceMode

    };

}
