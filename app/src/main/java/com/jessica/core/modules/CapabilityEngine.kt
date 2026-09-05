package com.jessica.core.modules

import android.content.Context


data class CapabilityResult(
    val success: Boolean,
    val message: String
)


class CapabilityEngine(
    context: Context
) {

    private val memoryStorage =
        MemoryStorage(context)

    private val eventStorage =
        EventStorage(context)


    fun execute(
        capability: String
    ): CapabilityResult {

        return when (capability) {


            "system_check" -> {

                eventStorage.saveEvent(
                    type = "system",
                    message = "Выполнена системная проверка"
                )

                CapabilityResult(
                    success = true,
                    message = "Системная проверка выполнена"
                )

            }


            "block_analysis" -> {

                eventStorage.saveEvent(
                    type = "analysis",
                    message = "Выполнен анализ блоков"
                )

                CapabilityResult(
                    success = true,
                    message = "Анализ блоков выполнен"
                )

            }


            "report_generation" -> {

                eventStorage.saveEvent(
                    type = "report",
                    message = "Сформирован отчёт"
                )

                CapabilityResult(
                    success = true,
                    message = "Отчёт сформирован"
                )

            }


            "list_blocks" -> {

                eventStorage.saveEvent(
                    type = "blocks",
                    message = "Запрошен список блоков"
                )

                CapabilityResult(
                    success = true,
                    message = "Получен список блоков"
                )

            }


            "check_status" -> {

                eventStorage.saveEvent(
                    type = "system",
                    message = "Проверен статус системы"
                )

                CapabilityResult(
                    success = true,
                    message = "Статус системы проверен"
                )

            }


            "manage_blocks" -> {

                eventStorage.saveEvent(
                    type = "blocks",
                    message = "Запрошено управление блоками"
                )

                CapabilityResult(
                    success = true,
                    message = "Управление блоками доступно"
                )

            }


            "create_reports" -> {

                eventStorage.saveEvent(
                    type = "report",
                    message = "Запрошено создание отчёта"
                )

                CapabilityResult(
                    success = true,
                    message = "Создание отчётов доступно"
                )

            }


            "save_reports" -> {

                eventStorage.saveEvent(
                    type = "report",
                    message = "Запрошено сохранение отчёта"
                )

                CapabilityResult(
                    success = true,
                    message = "Сохранение отчётов доступно"
                )

            }


            "load_reports" -> {

                eventStorage.saveEvent(
                    type = "report",
                    message = "Запрошена загрузка отчётов"
                )

                CapabilityResult(
                    success = true,
                    message = "Загрузка отчётов доступна"
                )

            }


            "analyze_results" -> {

                eventStorage.saveEvent(
                    type = "analysis",
                    message = "Выполнен анализ результатов"
                )

                CapabilityResult(
                    success = true,
                    message = "Анализ результатов выполнен"
                )

            }


            "save_memory" -> {

                memoryStorage.saveMemory(
                    key = "memory_core_test",
                    value = "Memory Core работает"
                )

                eventStorage.saveEvent(
                    type = "memory",
                    message = "Сохранена запись memory_core_test"
                )

                CapabilityResult(
                    success = true,
                    message =
                        "Память сохранена: memory_core_test"
                )

            }


            "load_memory" -> {

                val memory =
                    memoryStorage.getMemory(
                        "memory_core_test"
                    )


                if (memory != null) {

                    eventStorage.saveEvent(
                        type = "memory",
                        message = "Загружена запись memory_core_test"
                    )

                    CapabilityResult(
                        success = true,
                        message =
                            "Память загружена: ${memory.value}"
                    )

                } else {

                    eventStorage.saveEvent(
                        type = "error",
                        message = "Запись memory_core_test не найдена"
                    )

                    CapabilityResult(
                        success = false,
                        message =
                            "Запись memory_core_test не найдена"
                    )

                }

            }


            "save_events" -> {

                eventStorage.saveEvent(
                    type = "event",
                    message = "Тестовое событие Memory Core"
                )

                CapabilityResult(
                    success = true,
                    message = "Событие сохранено"
                )

            }


            "load_events" -> {

                val events =
                    eventStorage.loadEvents()


                CapabilityResult(
                    success = true,
                    message =
                        "Загружено событий: ${events.size}"
                )

            }


            "action_history" -> {

                val events =
                    eventStorage
                        .loadEvents()
                        .takeLast(10)


                if (events.isEmpty()) {

                    CapabilityResult(
                        success = true,
                        message = "История действий пока пуста"
                    )

                } else {

                    val history =
                        buildString {

                            appendLine(
                                "Последние действия:"
                            )

                            events.forEach { event ->

                                appendLine(
                                    "• ${event.type}: ${event.message}"
                                )

                            }

                        }


                    CapabilityResult(
                        success = true,
                        message = history
                    )

                }

            }


            else -> {

                eventStorage.saveEvent(
                    type = "error",
                    message =
                        "Неизвестная возможность: $capability"
                )

                CapabilityResult(
                    success = false,
                    message =
                        "Неизвестная возможность: $capability"
                )

            }

        }

    }

}
