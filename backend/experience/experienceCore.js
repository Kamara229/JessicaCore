/*
 * =========================================================
 * JESSICA EXPERIENCE CORE
 * =========================================================
 *
 * Центральный координатор накопленного опыта Jessica.
 *
 * В дальнейшем цепочка будет:
 *
 * task
 *   ↓
 * Experience Search
 *   ↓
 * подходящий Skill
 *   ↓
 * Experience Context
 *   ↓
 * PlanningContext
 *   ↓
 * Planner
 *
 *
 * Этот файл НЕ должен содержать:
 *
 * - хранение данных;
 * - AI-анализ обучения;
 * - поиск внутри базы;
 * - работу Planner;
 * - бизнес-логику Earnings.
 *
 * Он только объединяет Experience-модули
 * в единую рабочую цепочку.
 *
 * =========================================================
 */


/*
 * =========================================================
 * EMPTY EXPERIENCE RESULT
 * =========================================================
 */


function createEmptyExperienceResult() {

    return {

        found:
            false,

        experience:
            null,

        confidence:
            0,

        source:
            "none"

    };

}


/*
 * =========================================================
 * RESOLVE EXPERIENCE
 * =========================================================
 *
 * Главная точка входа Experience.
 *
 * Сейчас это безопасная базовая реализация.
 *
 * Следующим шагом сюда будет подключён
 * experienceSearch.js.
 *
 * =========================================================
 */


export async function resolveExperience(
    task
) {


    const cleanTask =
        String(
            task || ""
        ).trim();


    if (!cleanTask) {

        return createEmptyExperienceResult();

    }


    /*
     * Пока Experience Search
     * ещё не подключён.
     *
     * Jessica продолжает работать
     * через обычный Planner.
     */


    return createEmptyExperienceResult();

}
