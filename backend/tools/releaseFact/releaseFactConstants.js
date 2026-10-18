/*
 * =========================================================
 * JESSICA RELEASE FACT CONSTANTS
 * =========================================================
 */

export const MAX_CONTENT_LENGTH = 50000;

export const MAX_SECTION_LENGTH = 10000;

export const RELEASE_SIGNAL =
    /(latest|current|stable|release|released|version|releases|релиз|выпуск|верси|стабильн|актуальн|текущ)/iu;

export const PRERELEASE_SIGNAL =
    /(?:alpha|beta|preview|pre-release|prerelease|release\s+candidate|\brc\b|devel|development)/iu;

export const SEMVER_PATTERN =
    /\b(\d{1,3}(?:\.\d{1,4}){1,3})\b/g;

export const SINGLE_MAJOR_PATTERN =
    /\b(?:version|jdk|java\s+se|postgresql|python|rust|node(?:\.js)?)\s+(\d{1,3})\b/giu;

export const ISO_DATE_PATTERN =
    /\b(20\d{2})-(0[1-9]|1[0-2])-([0-2]\d|3[01])\b/g;

export const SUBJECT_STOPWORDS =
    new Set([
        "найди","найти","на","официальном","официальный","официального",
        "сайте","сайт","дату","дата","выпуска","выпуск","релиза","релиз",
        "актуальной","актуальная","актуальный","стабильной","стабильная",
        "версии","версия","проверить","проверенную","укажи","определи",
        "последнюю","последняя","текущую","текущая","затем","рассчитай",
        "сколько","полных","дней","прошло","между","используй","найденную",
        "первой","части","the","a","an","find","official","website","site",
        "latest","current","stable","version","release","released","date",
        "source","verify","check","determine","most","recent","provide",
        "and","or","of","to","from","with","for","in","on","at"
    ]);
