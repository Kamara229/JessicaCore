import {
    loadExperiences,
    disableExperience
} from "./supabaseExperienceStore.js";

import {
    loadExperienceHistory,
    loadExperienceVersion
} from "./supabaseExperienceHistoryStore.js";

import {
    saveExperienceAtomic
} from "./supabaseExperienceWriter.js";


/*
 * =========================================================
 * JESSICA EXPERIENCE STORAGE
 * =========================================================
 *
 * Центральный интерфейс хранения опыта Jessica.
 *
 *
 * Остальная система работает только через этот файл.
 *
 *
 * Experience Core
 *       ↓
 * experienceStorage.js
 *       ↓
 * ┌────────────────────────────────────────┐
 * │ supabaseExperienceStore.js             │
 * │ supabaseExperienceHistoryStore.js      │
 * │ supabaseExperienceWriter.js            │
 * └────────────────────────────────────────┘
 *       ↓
 * Supabase / PostgreSQL
 *
 *
 * Этот файл НЕ содержит:
 *
 * - SQL;
 * - Supabase Client;
 * - поиск подходящего Skill;
 * - Learning;
 * - Planner;
 * - Earnings.
 *
 *
 * Его задача:
 *
 * предоставить Experience единый
 * интерфейс работы с хранилищем.
 *
 * =========================================================
 */


/*
 * =========================================================
 * LOAD SKILLS
 * =========================================================
 *
 * Загружает все активные Skills Jessica.
 *
 * =========================================================
 */


export async function loadExperienceSkills() {


    try {


        const experiences =
            await loadExperiences();


        return Array.isArray(
            experiences
        )
            ? experiences
            : [];


    } catch (error) {


        console.error(
            "Experience Storage load error:",
            error
        );


        /*
         * Ошибка базы не должна
         * ломать всю Jessica.
         *
         * В этом случае первый слой
         * Experience будет просто пустым.
         */


        return [];


    }


}


/*
 * =========================================================
 * SAVE SKILL
 * =========================================================
 *
 * Сохраняет новую версию Skill.
 *
 * Используется только атомарная операция:
 *
 * History
 * +
 * Current Skill
 *
 * =========================================================
 */


export async function saveExperienceSkill(
    experience
) {


    return await saveExperienceAtomic(
        experience
    );


}


/*
 * =========================================================
 * DISABLE SKILL
 * =========================================================
 *
 * Skill не удаляется физически.
 *
 * Он остаётся в базе для:
 *
 * - истории;
 * - анализа;
 * - будущего восстановления;
 * - обучения.
 *
 * =========================================================
 */


export async function disableExperienceSkill(
    skillId
) {


    return await disableExperience(
        skillId
    );


}


/*
 * =========================================================
 * LOAD HISTORY
 * =========================================================
 */


export async function getExperienceHistory(
    skillId
) {


    try {


        const history =
            await loadExperienceHistory(
                skillId
            );


        return Array.isArray(
            history
        )
            ? history
            : [];


    } catch (error) {


        console.error(
            "Experience History load error:",
            error
        );


        return [];


    }


}


/*
 * =========================================================
 * LOAD SPECIFIC VERSION
 * =========================================================
 */


export async function getExperienceVersion(

    skillId,

    version

) {


    try {


        return await loadExperienceVersion(

            skillId,

            version

        );


    } catch (error) {


        console.error(
            "Experience Version load error:",
            error
        );


        return null;


    }


}
