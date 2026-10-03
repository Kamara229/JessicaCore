/*
 * =========================================================
 * JESSICA EXPERIENCE ANALYZER v5
 * =========================================================
 *
 * Главный координатор анализа Experience.
 *
 *
 * Flow:
 *
 * Execution Trace
 *        ↓
 * Trace Validation
 *        ↓
 * Learning Evidence
 *        ↓
 *
 * ┌─────────────────────────────────────┐
 * │ Existing Experience was used?       │
 * └─────────────────────────────────────┘
 *        │
 *       YES
 *        ↓
 * SKILL_IMPROVEMENT
 *
 *
 * Если Existing Experience не использовался:
 *
 * Execution Trace
 *        ↓
 * Known Pattern Matcher
 *        ↓
 * NEW_SKILL Candidate
 *
 *
 * Если известный Pattern не найден:
 *
 * DISCOVERY REQUIRED
 *
 * На текущем этапе такой Trace
 * НЕ превращается автоматически
 * в Skill эвристикой.
 *
 * Следующий этап архитектуры:
 *
 * AI Pattern Extraction
 *        ↓
 * Grounding
 *        ↓
 * Generalization
 *        ↓
 * Dynamic NEW_SKILL Candidate
 *
 *
 * Ответственность:
 *
 * - принять Execution Trace;
 * - определить качество Execution;
 * - собрать Learning Metrics;
 * - определить использование Existing Skill;
 * - определить известный Pattern;
 * - выбрать Learning Action;
 * - сформировать Skill Candidate;
 * - сохранить диагностический контекст анализа.
 *
 *
 * НЕ:
 *
 * - сохраняет Experience;
 * - работает с Supabase;
 * - принимает AUTO_APPROVE;
 * - управляет версиями;
 * - накапливает KEEP_CANDIDATE;
 * - вызывает AI;
 * - самостоятельно придумывает
 *   новый Skill без Grounding.
 *
 * =========================================================
 */


import {
    matchExperiencePattern
} from "./experiencePatternMatcher.js";


import {
    getOccurrences,
    getExamples,
    calculateSuccessRate,
    calculateExperienceConfidence,
    calculateExperienceMaturity,
    resolveExperienceMaturityLevel
} from "./experienceConfidence.js";


import {
    buildNewSkillCandidate,
    buildSkillImprovementCandidate
} from "./experienceCandidateBuilder.js";





/*
 * =========================================================
 * NORMALIZE
 * =========================================================
 */


function isObject(
    value
) {

    return (

        value &&

        typeof value === "object" &&

        !Array.isArray(value)

    );

}





function normalizeNumber(
    value
) {

    const number =
        Number(value);


    return Number.isFinite(number)

        ? number

        : 0;

}





function normalizeText(
    value
) {

    return String(
        value || ""
    )
    .trim();

}





/*
 * =========================================================
 * IGNORE RESULT
 * =========================================================
 */


function buildIgnoreResult({

    reason,

    analysisType = "NONE",

    discoveryRequired = false,

    metrics = null,

    metadata = {}

} = {}) {


    return {


        action:
            "IGNORE",



        reusable:
            false,



        reason:

            normalizeText(
                reason
            )

            ||

            "Нет пригодного опыта для обучения",



        skillCandidate:
            null,



        analysisType,



        discoveryRequired:

            discoveryRequired === true,



        metrics:

            metrics || null,



        metadata:

            isObject(
                metadata
            )

                ? metadata

                : {}

    };

}





/*
 * =========================================================
 * TRACE SUCCESS
 * =========================================================
 *
 * Learning Analyzer не должен зависеть
 * только от trace.stats.completed.
 *
 * Execution Trace сейчас существует
 * в нескольких представлениях.
 *
 * Поэтому принимаем несколько
 * подтверждённых признаков успеха.
 *
 * =========================================================
 */


function isSuccessfulTrace(
    trace
) {


    /*
     * Terminal Result
     */


    if(
        typeof trace?.result?.success ===
        "boolean"
    ){

        return trace.result.success;

    }



    /*
     * Direct flag
     */


    if(
        typeof trace?.success ===
        "boolean"
    ){

        return trace.success;

    }



    /*
     * Legacy / outer stats
     */


    const completed =

        normalizeNumber(

            trace?.stats?.completed

            ??

            trace?.statistics?.completed

        );



    if(
        completed > 0
    ){

        return true;

    }



    /*
     * Context / Result status
     */


    const status =

        normalizeText(
            trace?.status
        )
        .toUpperCase();



    if(
        status === "COMPLETED"
    ){

        return true;

    }



    return false;

}





/*
 * =========================================================
 * BUILD LEARNING METRICS
 * =========================================================
 */


function buildLearningMetrics({

    trace,

    matchScore

}) {


    const occurrences =

        getOccurrences(
            trace
        );



    const examples =

        getExamples(
            trace
        );



    const successRate =

        calculateSuccessRate(
            examples
        );



    const confidence =

        calculateExperienceConfidence({

            matchScore,

            successRate,

            occurrences

        });



    const maturity =

        calculateExperienceMaturity(
            occurrences
        );



    const maturityLevel =

        resolveExperienceMaturityLevel(
            occurrences
        );



    return {


        occurrences,



        examples,



        successRate,



        confidence,



        maturity,



        maturityLevel,



        matchScore:

            Math.max(

                0,

                Math.min(

                    1,

                    normalizeNumber(
                        matchScore
                    )

                )

            )

    };

}





/*
 * =========================================================
 * EXPERIENCE USAGE
 * =========================================================
 */


function getExperienceUsage(
    trace
) {


    if(
        !isObject(
            trace?.experienceUsage
        )
    ){

        return null;

    }



    return trace.experienceUsage;

}





function getUsedSkills(
    trace
) {


    const usage =

        getExperienceUsage(
            trace
        );



    if(
        !usage ||
        usage.used !== true
    ){

        return [];

    }



    if(
        !Array.isArray(
            usage.skills
        )
    ){

        return [];

    }



    return usage.skills.filter(

        item =>
            item &&
            typeof item === "object"

    );

}





/*
 * =========================================================
 * ANALYZE EXISTING SKILL
 * =========================================================
 *
 * Если Existing Experience реально
 * участвовал в Execution,
 * новый успешный Execution является
 * дополнительным evidence.
 *
 *
 * На текущем этапе:
 *
 * SKILL_IMPROVEMENT
 * =
 * EVIDENCE_REINFORCEMENT
 *
 *
 * Семантическое изменение workflow,
 * constraints и validationRules
 * будет отдельным следующим слоем:
 *
 * Experience Improvement Analyzer.
 *
 * =========================================================
 */


function analyzeExistingSkill(
    trace
) {


    const skills =

        getUsedSkills(
            trace
        );



    if(
        skills.length === 0
    ){

        return null;

    }



    const metrics =

        buildLearningMetrics({

            trace,

            /*
             * Existing Experience уже был
             * найден и применён.
             *
             * Поэтому соответствие Skill
             * текущему execution считаем
             * максимальным.
             */

            matchScore:
                1

        });



    const skillCandidate =

        buildSkillImprovementCandidate({

            skills,

            trace,

            confidence:
                metrics.confidence,

            maturity:
                metrics.maturity,

            occurrences:
                metrics.occurrences

        });



    if(
        !skillCandidate
    ){

        return buildIgnoreResult({

            reason:
                "Не удалось сформировать Improvement Candidate",

            analysisType:
                "EXISTING_SKILL",

            metrics,

            metadata: {

                usedSkills:
                    skills.length

            }

        });

    }



    return {


        action:

            "SKILL_IMPROVEMENT",



        reusable:

            true,



        reason:

            "Существующий Experience Skill получил новый подтверждённый опыт",



        skillCandidate,



        analysisType:

            "EXISTING_SKILL",



        improvementType:

            "EVIDENCE_REINFORCEMENT",



        metrics,



        metadata: {


            experienceUsed:
                true,


            experienceSource:

                normalizeText(
                    trace
                        ?.experienceUsage
                        ?.source
                )

                ||

                null,


            usedSkills:
                skills.length,


            targetSkillId:

                skillCandidate.targetSkillId

                ||

                skills[0]?.id

                ||

                skills[0]?.skillId

                ||

                null

        }

    };

}





/*
 * =========================================================
 * ANALYZE KNOWN PATTERN
 * =========================================================
 *
 * Использует только seed / known patterns.
 *
 * Это быстрый deterministic путь.
 *
 * Он НЕ является пределом будущего
 * самообучения Jessica.
 *
 * =========================================================
 */


function analyzeKnownPattern(
    trace
) {


    const matched =

        matchExperiencePattern(
            trace?.task
        );



    if(
        !matched ||
        !matched.pattern
    ){

        return null;

    }



    const metrics =

        buildLearningMetrics({

            trace,

            matchScore:

                matched.matchScore

        });



    const skillCandidate =

        buildNewSkillCandidate({

            pattern:

                matched.pattern,



            trace,



            confidence:

                metrics.confidence,



            maturity:

                metrics.maturity,



            occurrences:

                metrics.occurrences

        });



    if(
        !skillCandidate
    ){

        return buildIgnoreResult({

            reason:
                "Known Pattern найден, но Candidate не сформирован",

            analysisType:
                "KNOWN_PATTERN",

            metrics,

            metadata: {

                patternId:

                    matched
                        ?.pattern
                        ?.id

                    ||

                    null

            }

        });

    }



    return {


        action:

            "NEW_SKILL",



        reusable:

            true,



        reason:

            "Обнаружен известный повторяемый сценарий, пригодный для Experience Skill",



        skillCandidate,



        analysisType:

            "KNOWN_PATTERN",



        metrics,



        metadata: {


            patternId:

                matched
                    ?.pattern
                    ?.id

                ||

                null,


            patternName:

                matched
                    ?.pattern
                    ?.name

                ||

                null,


            matchedKeywords:

                Array.isArray(
                    matched.matchedKeywords
                )

                    ?

                    matched.matchedKeywords

                    :

                    [],


            matchScore:

                metrics.matchScore

        }

    };

}





/*
 * =========================================================
 * BUILD DISCOVERY REQUIRED RESULT
 * =========================================================
 *
 * Успешный Execution есть,
 * Existing Skill не использовался,
 * Known Pattern не найден.
 *
 *
 * Такой Execution НЕ должен просто
 * исчезать как "нечему учиться".
 *
 * Он является кандидатом на
 * самостоятельное открытие нового Skill.
 *
 *
 * Пока AI Pattern Extractor
 * ещё не подключён, action остаётся IGNORE,
 * чтобы не создавать ложные Skills
 * эвристикой.
 *
 *
 * Но discoveryRequired=true позволяет
 * следующему этапу архитектуры явно
 * отличить:
 *
 * "обучаться нечему"
 *
 * от
 *
 * "нужен новый Pattern Discovery".
 *
 * =========================================================
 */


function buildDiscoveryRequiredResult(
    trace
) {


    const metrics =

        buildLearningMetrics({

            trace,

            /*
             * Pattern ещё неизвестен,
             * поэтому match quality
             * пока не определён.
             */

            matchScore:
                0

        });



    return buildIgnoreResult({

        reason:

            "Успешный Execution не соответствует известному Experience Pattern; требуется самостоятельное извлечение нового Pattern",



        analysisType:

            "PATTERN_DISCOVERY_REQUIRED",



        discoveryRequired:

            true,



        metrics,



        metadata: {


            task:

                normalizeText(
                    trace?.task
                ),


            traceId:

                trace?.id ||

                null,


            existingExperienceUsed:

                false

        }

    });

}





/*
 * =========================================================
 * MAIN ANALYSIS
 * =========================================================
 */


export function analyzeExecutionTrace(
    trace
) {


    /*
     * =====================================================
     * 1. VALIDATE TRACE
     * =====================================================
     */


    if(
        !isObject(
            trace
        )
    ){

        return buildIgnoreResult({

            reason:
                "Execution Trace отсутствует",

            analysisType:
                "INVALID_TRACE"

        });

    }



    /*
     * =====================================================
     * 2. SUCCESS CHECK
     * =====================================================
     *
     * Пока Learning создаёт и улучшает
     * Experience только из успешных
     * executions.
     *
     *
     * Failure Learning будет отдельным
     * следующим каналом:
     *
     * failurePatterns
     * avoidPatterns
     * constraints
     *
     * =====================================================
     */


    if(
        !isSuccessfulTrace(
            trace
        )
    ){

        return buildIgnoreResult({

            reason:
                "Execution не подтверждён как успешный",

            analysisType:
                "UNSUCCESSFUL_EXECUTION",

            metadata: {

                failureLearningRequired:
                    true

            }

        });

    }



    /*
     * =====================================================
     * 3. EXISTING EXPERIENCE
     * =====================================================
     *
     * Existing Skill всегда имеет
     * приоритет над созданием нового.
     *
     * Если Jessica уже использовала Skill,
     * новый Execution должен развивать
     * именно эту линию Experience.
     *
     * =====================================================
     */


    const existing =

        analyzeExistingSkill(
            trace
        );



    if(
        existing
    ){

        return existing;

    }



    /*
     * =====================================================
     * 4. KNOWN PATTERN
     * =====================================================
     */


    const knownPattern =

        analyzeKnownPattern(
            trace
        );



    if(
        knownPattern
    ){

        return knownPattern;

    }



    /*
     * =====================================================
     * 5. UNKNOWN SUCCESSFUL EXPERIENCE
     * =====================================================
     *
     * Раньше здесь было:
     *
     * "Повторяемый сценарий не найден"
     * → IGNORE
     *
     *
     * Теперь такой Execution явно
     * помечается как требующий
     * Pattern Discovery.
     *
     * Следующий этап:
     *
     * AI Experience Pattern Extractor.
     *
     * =====================================================
     */


    return buildDiscoveryRequiredResult(
        trace
    );

                }
