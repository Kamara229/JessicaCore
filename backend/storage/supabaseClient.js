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


    const secretKey =
        String(
            process.env.SUPABASE_SECRET_KEY || ""
        ).trim();


    return {

        url,

        secretKey

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

        config.secretKey

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
     * =====================================================
     * CONFIGURATION CHECK
     * =====================================================
     */


    const config =
        getSupabaseConfig();


    if (
        !config.url ||
        !config.secretKey
    ) {

        throw new Error(
            "Supabase не настроен: отсутствуют SUPABASE_URL или SUPABASE_SECRET_KEY"
        );

    }


    /*
     * =====================================================
     * CREATE CLIENT
     * =====================================================
     *
     * Secret key используется только backend.
     *
     * Он НЕ должен попадать:
     *
     * - в Android;
     * - в GitHub;
     * - в ответы API;
     * - в логи.
     *
     * =====================================================
     */


    supabaseClient =
        createClient(

            config.url,

            config.secretKey,

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
