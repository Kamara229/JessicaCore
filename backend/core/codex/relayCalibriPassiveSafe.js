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
    return explicitTest.test(text)||openTestChat.test(text)||endCommand.test(text) ? null : status;
}
