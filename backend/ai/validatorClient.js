import {
    getGroqClient
} from "./groqClient.js";


const VALIDATOR_MODEL =
    "llama-3.1-8b-instant";



export async function validatorChat(
    messages
) {


    const groq =
        getGroqClient();


    return await groq
        .chat
        .completions
        .create({

            model:
                VALIDATOR_MODEL,

            temperature:
                0,

            messages

        });

}
