import {
    getGroqClient
} from "./groqClient.js";


const PLANNER_MODEL =
    "openai/gpt-oss-20b";



export async function plannerChat(
    messages
) {


    const groq =
        getGroqClient();


    return await groq
        .chat
        .completions
        .create({

            model:
                PLANNER_MODEL,

            temperature:
                0,

            messages

        });

}
