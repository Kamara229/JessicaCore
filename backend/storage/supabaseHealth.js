import {
    getSupabaseClient,
    isSupabaseConfigured
} from "./supabaseClient.js";


/*
 * =========================================================
 * JESSICA SUPABASE HEALTH
 * =========================================================
 *
 * Проверяет:
 *
 * - настроен ли Supabase;
 * - доступна ли база;
 * - существуют ли таблицы Experience.
 *
 *
 * Этот модуль НЕ:
 *
 * - хранит Experience;
 * - изменяет данные;
 * - создаёт таблицы;
 * - выводит секретные ключи;
 * - содержит бизнес-логику.
 *
 * =========================================================
 */


const EXPERIENCE_TABLE =
    "jessica_experience_skills";


const EXPERIENCE_VERSIONS_TABLE =
    "jessica_experience_skill_versions";


/*
 * =========================================================
 * TEST TABLE
 * =========================================================
 *
 * Проверяем существование таблицы
 * и возможность чтения.
 *
 * Не привязываемся к конкретным колонкам,
 * потому что таблицы могут иметь
 * разную структуру.
 *
 * =========================================================
 */


async function testTable(
    supabase,
    tableName
) {


    const {
        error
    } =
        await supabase
            .from(
                tableName
            )
            .select(
                "*"
            )
            .limit(
                1
            );


    if (error) {

        throw new Error(
            `Таблица ${tableName} недоступна: ${error.message}`
        );

    }


    return true;


}


/*
 * =========================================================
 * TEST SUPABASE CONNECTION
 * =========================================================
 */


export async function testSupabaseConnection() {


    /*
     * =====================================================
     * CONFIGURATION
     * =====================================================
     */


    if (
        !isSupabaseConfigured()
    ) {

        return {

            configured:
                false,

            connected:
                false,

            experienceTables:
                false,

            text:
                "Supabase не настроен"

        };

    }


    /*
     * =====================================================
     * CONNECTION
     * =====================================================
     */


    try {


        const supabase =
            getSupabaseClient();


        /*
         * =================================================
         * CURRENT EXPERIENCE TABLE
         * =================================================
         */


        await testTable(
            supabase,
            EXPERIENCE_TABLE
        );


        /*
         * =================================================
         * EXPERIENCE HISTORY TABLE
         * =================================================
         */


        await testTable(
            supabase,
            EXPERIENCE_VERSIONS_TABLE
        );


        /*
         * =================================================
         * SUCCESS
         * =================================================
         */


        return {

            configured:
                true,

            connected:
                true,

            experienceTables:
                true,

            text:
                "Supabase подключён"

        };


    } catch (error) {


        /*
         * Полную техническую ошибку
         * оставляем только в логах backend.
         *
         * Секретные ключи
         * никогда не выводятся.
         */


        console.error(
            "Supabase health error:",
            error
        );


        return {

            configured:
                true,

            connected:
                false,

            experienceTables:
                false,

            text:
                "Не удалось подключиться к Supabase"

        };


    }


}
