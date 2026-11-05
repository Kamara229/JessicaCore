import {calibriPassiveRelayHandoff} from "./relayCalibriPassiveHandoff.js";

/**
 * Unicode-aware safety guard layered on the deterministic no-test facade.
 * JS word-boundary escapes are ASCII-based for Cyrillic, so a Russian
 * imperative followed by "тест" must use a Unicode-aware lookahead.
 *
 * A passive classification is never permission to run a test or stop a
 * session. Explicit owner instructions remain with the main interpreter.
 */
export function safeCalibriPassiveHandoff(message) {
    const status=calibriPassiveRelayHandoff(message);
    if (status===null) return null;
    const text=String(message).replace(/\s+/gu," ");
    const explicitTest=/(?:запусти|выполни|проведи|повтори|отправь)\s+(?:(?:новый|следующий|этот|один)\s+)?тест(?=$|[^\p{L}\p{N}])/iu;
    const openTestChat=/(?:создай|открой)\s+(?:новый\s+)?тестовый\s+чат(?=$|[^\p{L}\p{N}])/iu;
    const endCommand=/(?:^|[^\p{L}\p{N}])(?:план\s+(?:полностью\s+)?заверш[её]н|заверши\s+сессию)(?=$|[^\p{L}\p{N}])/iu;
    // A command to send a concrete prompt into a Jessica chat has priority,
    // even when adjacent sentences say that ordinary CALIBRI QA is GitHub-only.
    const ordinaryChatTest=/(?:в\\s+(?:обычном\\s+)?чате\\s+(?:Jessica|Джессик[иауы])\\s+(?:выполни|отправь|проверь)|(?:выполни|отправь|проведи)\\s+(?:в\\s+)?(?:обычный\\s+)?чат\\s+(?:Jessica|Джессик[иауы])|Точная\\s+формулировка\\s*[:：]|Задача\\s+для\\s+(?:Jessica|Джессик[иауы])\\s*[:：])/iu;
    return explicitTest.test(text)||openTestChat.test(text)||ordinaryChatTest.test(text)||endCommand.test(text) ? null : status;
}
