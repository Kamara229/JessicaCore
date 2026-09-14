/*
 * =========================================================
 * JESSICA LEARNING ANALYZER SCHEMA
 * =========================================================
 *
 * JSON Schema ответа Learning Analyzer.
 *
 *
 * Используется Groq Structured Outputs:
 *
 * response_format:
 * json_schema
 * strict: true
 *
 *
 * Это гарантирует:
 *
 * - корректный JSON;
 * - наличие обязательных полей;
 * - правильные типы данных;
 * - отсутствие неожиданных полей.
 *
 *
 * Смысл и безопасность результата
 * по-прежнему проверяют:
 *
 * - Learning Analyzer Validator;
 * - Grounding Validator;
 * - Generalization Validator.
 *
 *
 * Schema НЕ:
 *
 * - проверяет смысл правил;
 * - проверяет evidence по correction;
 * - сохраняет Experience;
 * - подтверждает Learning.
 *
 * =========================================================
 */


export const LEARNING_ANALYZER_RESPONSE_FORMAT = {

    type:
        "json_schema",

    json_schema: {

        name:
            "jessica_learning_analysis",

        strict:
            true,

        schema: {

            type:
                "object",

            properties: {


                /*
                 * =========================================
                 * ANALYSIS
                 * =========================================
                 */


                analysis: {

                    type:
                        "object",

                    properties: {

                        errorSummary: {
                            type:
                                "string"
                        },

                        correctionSummary: {
                            type:
                                "string"
                        },

                        differenceSummary: {
                            type:
                                "string"
                        }

                    },

                    required: [

                        "errorSummary",

                        "correctionSummary",

                        "differenceSummary"

                    ],

                    additionalProperties:
                        false

                },


                /*
                 * =========================================
                 * UNDERSTANDING
                 * =========================================
                 */


                understanding: {

                    type:
                        "string"

                },


                /*
                 * =========================================
                 * REUSABLE
                 * =========================================
                 */


                reusable: {

                    type:
                        "boolean"

                },


                /*
                 * =========================================
                 * CLARIFICATION QUESTIONS
                 * =========================================
                 */


                clarificationQuestions: {

                    type:
                        "array",

                    items: {

                        type:
                            "string"

                    }

                },


                /*
                 * =========================================
                 * UNSUPPORTED SUGGESTIONS
                 * =========================================
                 */


                unsupportedSuggestions: {

                    type:
                        "array",

                    items: {

                        type:
                            "string"

                    }

                },


                /*
                 * =========================================
                 * PROPOSED EXPERIENCE
                 * =========================================
                 */


                proposedExperience: {

                    type:
                        "object",

                    properties: {

                        name: {
                            type:
                                "string"
                        },

                        description: {
                            type:
                                "string"
                        },

                        taskTypes: {

                            type:
                                "array",

                            items: {
                                type:
                                    "string"
                            }

                        },

                        keywords: {

                            type:
                                "array",

                            items: {
                                type:
                                    "string"
                            }

                        },

                        tags: {

                            type:
                                "array",

                            items: {
                                type:
                                    "string"
                            }

                        },

                        strategy: {

                            type:
                                "array",

                            items: {
                                type:
                                    "string"
                            }

                        },

                        sourcePriority: {

                            type:
                                "array",

                            items: {
                                type:
                                    "string"
                            }

                        },

                        validationRules: {

                            type:
                                "array",

                            items: {
                                type:
                                    "string"
                            }

                        },

                        failurePatterns: {

                            type:
                                "array",

                            items: {
                                type:
                                    "string"
                            }

                        }

                    },

                    required: [

                        "name",

                        "description",

                        "taskTypes",

                        "keywords",

                        "tags",

                        "strategy",

                        "sourcePriority",

                        "validationRules",

                        "failurePatterns"

                    ],

                    additionalProperties:
                        false

                },


                /*
                 * =========================================
                 * GROUNDING EVIDENCE
                 * =========================================
                 */


                groundingEvidence: {

                    type:
                        "array",

                    items: {

                        type:
                            "object",

                        properties: {

                            field: {

                                type:
                                    "string",

                                enum: [

                                    "strategy",

                                    "sourcePriority",

                                    "validationRules",

                                    "failurePatterns"

                                ]

                            },

                            value: {
                                type:
                                    "string"
                            },

                            evidence: {
                                type:
                                    "string"
                            }

                        },

                        required: [

                            "field",

                            "value",

                            "evidence"

                        ],

                        additionalProperties:
                            false

                    }

                }

            },


            /*
             * =============================================
             * REQUIRED ROOT FIELDS
             * =============================================
             */


            required: [

                "analysis",

                "understanding",

                "reusable",

                "clarificationQuestions",

                "unsupportedSuggestions",

                "proposedExperience",

                "groundingEvidence"

            ],

            additionalProperties:
                false

        }

    }

};
