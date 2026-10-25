import {hasForeignEnglishProse, splitCodeSegments} from "./codeAwareLocale.js";
import {isVerifiedRequestedOriginalQuote} from "./verifiedQuotePreservation.js";

/**
 * Last-resort presentation contract.
 * Never silently return an English narrative under Russian locale if
 * localization failed. Keep source code byte-for-byte (never "translate" it).
 * This does not validate the program or alter its underlying execution trace.
 */
export function enforcePresentationLanguage(result,{task="",language="ru",override=false}={}) {
    if (language!=="ru" || override || !result || typeof result.text!=="string" ||
        !hasForeignEnglishProse(result.text)) return result;
    if (result.mode==="translation") return result;
    const code=splitCodeSegments(result.text).filter(x=>x.kind==="code").map(x=>x.text);
    if (code.length) return {
        ...result,
        success:false,
        partial:true,
        text:"Не удалось подготовить все пояснения на русском языке. " +
            "Исходный код сохранён без изменений; его выполнение и корректность не подтверждены. " +
            "Повторите запрос для полноценного ответа.\n\n"+code.join("\n\n"),
        outputLanguage:"ru",
        localizationStatus:"ru-narrative-not-verified"
    };
    return {
        ...result,
        success:false,
        partial:true,
        text:"Не удалось подготовить ответ на русском языке. " +
            "Англоязычный текст не был показан как готовый ответ. Повторите запрос.",
        outputLanguage:"ru",
        localizationStatus:"ru-narrative-not-verified"
    };
}
