import {
    createClient
} from "@supabase/supabase-js";


/*
 * =========================================================
 * JESSICA SUPABASE CLIENT
 * =========================================================
 *
 * Общая точка подключения backend Jessica
 * к Supabase / PostgreSQL.
 *
 *
 * Этот модуль будет использоваться:
 *
 * - Experience;
 * - Learning;
 * - Earnings;
 * - статистикой;
 * - другими будущими модулями.
 *
 *
 * ВАЖНО:
 *
 * Здесь НЕ должно быть:
 *
 * - логики Experience;
 * - логики Earnings;
 * - SQL конкретных модулей;
 * - бизнес-логики.
 *
 *
 * Этот файл отвечает только за:
 *
 * конфигурацию
 *      ↓
 * создание Supabase Client
 *      ↓
 * предоставление клиента другим модулям
 *
 * =========================================================
 */


/*
 * =========================================================
 * CLIENT CACHE
 * =========================================================
 *
 * Клиент создаётся только один раз.
 *
 * =========================================================
 */


let supabaseClient =
    null;


/*
 * =========================================================
 * CONFIGURATION
 * =========================================================
 */


function getSupabaseConfig() {

    const url =
        String(
            process.env.SUPABASE_URL || ""
        ).trim();


    const serviceRoleKey =
        String(
            process.env.SUPABASE_SERVICE_ROLE_KEY || ""
        ).trim();


    return {

        url,

        serviceRoleKey

    };

}


/*
 * =========================================================
 * CONFIGURED
 * =========================================================
 */


export function isSupabaseConfigured() {

    const config =
        getSupabaseConfig();


    return Boolean(

        config.url &&

        config.serviceRoleKey

    );

}


/*
 * =========================================================
 * GET CLIENT
 * =========================================================
 */


export function getSupabaseClient() {


    /*
     * Уже создан.
     */


    if (supabaseClient) {

        return supabaseClient;

    }


    /*
     * Проверяем конфигурацию.
     */


    const config =
        getSupabaseConfig();


    if (
        !config.url ||
        !config.serviceRoleKey
    ) {

        throw new Error(
            "Supabase не настроен: отсутствуют SUPABASE_URL или SUPABASE_SERVICE_ROLE_KEY"
        );

    }


    /*
     * Создаём единый backend-клиент.
     */


    supabaseClient =
        createClient(

            config.url,

            config.serviceRoleKey,

            {

                auth: {

                    persistSession:
                        false,

                    autoRefreshToken:
                        false

                }

            }

        );


    return supabaseClient;

}
