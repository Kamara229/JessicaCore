/*
 * =========================================================
 * JESSICA MEMORY RETRIEVER
 * =========================================================
 *
 * Поиск накопленного опыта Jessica.
 *
 *
 * Flow:
 *
 * User Task
 *      ↓
 * Memory Retriever
 *      ↓
 * Experience Skills
 *      ↓
 * Planner
 *
 *
 * НЕ:
 *
 * - создаёт Experience;
 * - обновляет память;
 * - принимает Learning решения.
 *
 * =========================================================
 */


import {
    getSupabaseClient
} from "../../storage/supabaseClient.js";





const TABLE_NAME =
    "experience_skills";





/*
 * =========================================================
 * NORMALIZE QUERY
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
 * SEARCH SKILLS
 * =========================================================
 */


export async function retrieveRelevantExperience(
    task,
    limit = 5
) {


    const query =
        normalizeText(
            task
        );



    if (
        !query
    ) {


        return {

            success:
                false,

            skills:
                [],

            reason:
                "Empty task"

        };

    }





    try {


        const {
            data,
            error
        } =
            await getSupabaseClient()

                .from(
                    TABLE_NAME
                )

                .select("*")

                .limit(
                    limit
                );





        if (
            error
        ) {


            return {


                success:
                    false,


                skills:
                    [],


                error:
                    error.message


            };

        }





        /*
         * Пока простой поиск.
         *
         * Следующим этапом заменим
         * на embeddings + vector search.
         */


        const skills =
            (data || [])

                .filter(

                    item => {


                        const text =

                            JSON.stringify(
                                item
                            )
                            .toLowerCase();



                        return text.includes(
                            query
                        );


                    }

                );





        return {


            success:
                true,


            skills



        };



    } catch(error) {


        return {


            success:
                false,


            skills:
                [],


            error:
                error.message


        };


    }


}
