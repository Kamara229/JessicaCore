/*
 * =========================================================
 * JESSICA EXPERIENCE RETRIEVAL
 * =========================================================
 *
 * Поиск похожего прошлого опыта.
 *
 *
 * НЕ:
 *
 * - выполняет задачи;
 * - создаёт Skill;
 * - изменяет память.
 *
 * Только извлекает знания.
 *
 * =========================================================
 */



/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function normalize(
    text
) {

    return String(text || "")
        .toLowerCase()
        .trim();

}





/*
 * =========================================================
 * CALCULATE SIMILARITY
 * =========================================================
 */


function calculateSimilarity(
    task,
    experience
) {


    const source =
        normalize(task);


    const examples =
        experience?.examples || [];



    let score =
        0;



    for (
        const example
        of examples
    ) {


        const exampleText =
            normalize(
                example.task
            );



        if (
            source.includes(exampleText) ||
            exampleText.includes(source)
        ) {

            score += 1;

        }


        const words =
            source.split(" ");



        for (
            const word
            of words
        ) {

            if (
                word.length > 4 &&
                exampleText.includes(word)
            ) {

                score += 0.1;

            }

        }


    }



    return Math.min(
        score / 2,
        1
    );

}





/*
 * =========================================================
 * RETRIEVE EXPERIENCE
 * =========================================================
 */


export function retrieveExperience(
    task,
    experiences = []
) {


    const results =
        experiences
            .map(

                item => ({

                    experience:item,

                    similarity:
                        calculateSimilarity(
                            task,
                            item
                        )

                })

            )
            .filter(

                item =>
                    item.similarity > 0

            )
            .sort(

                (a,b) =>
                    b.similarity -
                    a.similarity

            );



    return {


        found:
            results.length > 0,


        matches:
            results.slice(
                0,
                5
            ),


        best:

            results[0] ||
            null


    };

}
