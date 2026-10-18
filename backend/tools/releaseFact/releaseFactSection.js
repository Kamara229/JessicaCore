/*
 * =========================================================
 * JESSICA RELEASE FACT SECTION
 * =========================================================
 */

import {
    MAX_SECTION_LENGTH
} from "./releaseFactConstants.js";


function isReleaseHeading(
    line
) {

    const value =

        String(
            line || ""
        )
            .replace(
                /^#{1,6}\s*/,
                ""
            )
            .trim();


    return /^(latest\s+releases?|current\s+releases?|release\s+history|releases?|последн\p{L}*\s+релиз\p{L}*|актуальн\p{L}*\s+релиз\p{L}*|релиз\p{L}*)$/iu
        .test(
            value
        );

}


function isMarkdownHeading(
    line
) {

    return /^#{1,6}\s+\S/u.test(
        String(
            line || ""
        )
    );

}


export function extractPreferredReleaseSection(
    text
) {

    const lines =

        text.split(
            "\n"
        );


    let offset = 0;


    for(
        let index = 0;
        index < lines.length;
        index++
    ){

        const line =
            lines[index];


        if(
            !isReleaseHeading(
                line
            )
        ){

            offset +=
                line.length + 1;


            continue;

        }


        const start =
            offset;


        let end =
            Math.min(
                text.length,
                start + MAX_SECTION_LENGTH
            );


        let cursor =
            offset + line.length + 1;


        for(
            let next = index + 1;
            next < lines.length;
            next++
        ){

            const nextLine =
                lines[next];


            if(
                isMarkdownHeading(
                    nextLine
                )
            ){

                end =
                    Math.min(
                        end,
                        cursor
                    );


                break;

            }


            cursor +=
                nextLine.length + 1;

        }


        return {

            text:

                text.slice(
                    start,
                    end
                ),

            start,

            preferred:
                true

        };

    }


    return {

        text,

        start:
            0,

        preferred:
            false

    };

}
