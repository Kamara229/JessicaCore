package com.jessica.core.modules


data class BlockTestResult(
    val passed: Boolean,
    val report: String
)


class BlockTester {


    fun test(
        block: Block,
        manager: BlockManager
    ): BlockTestResult {


        val checks =
            mutableListOf<String>()


        val idCheck =
            block.id.isNotBlank()

        checks.add(
            "ID блока: " +
                if (idCheck) "OK"
                else "ОШИБКА"
        )


        val nameCheck =
            block.name.isNotBlank()

        checks.add(
            "Имя блока: " +
                if (nameCheck) "OK"
                else "ОШИБКА"
        )


        val versionCheck =
            block.version.isNotBlank()

        checks.add(
            "Версия: " +
                if (versionCheck) "OK"
                else "ОШИБКА"
        )


        val authorCheck =
            block.author.isNotBlank()

        checks.add(
            "Автор: " +
                if (authorCheck) "OK"
                else "ОШИБКА"
        )


        val typeCheck =
            block.type.isNotBlank() &&
            block.type != "unknown"

        checks.add(
            "Тип блока: " +
                if (typeCheck) "OK"
                else "ОШИБКА"
        )


        val statusCheck =
            block.status == "ACTIVE"

        checks.add(
            "Статус: " +
                if (statusCheck) "OK"
                else "ОШИБКА"
        )


        val descriptionCheck =
            block.description.isNotBlank()

        checks.add(
            "Описание: " +
                if (descriptionCheck) "OK"
                else "ОШИБКА"
        )


        val capabilitiesCheck =
            block.capabilities.isNotEmpty()

        checks.add(
            "Возможности: " +
                if (capabilitiesCheck) "OK"
                else "ОШИБКА"
        )


        val capabilitiesValidCheck =
            block.capabilities.all {
                it.isNotBlank()
            }

        checks.add(
            "Формат возможностей: " +
                if (capabilitiesValidCheck) "OK"
                else "ОШИБКА"
        )


        val installed =
            manager
                .getBlocks()
                .any {
                    it.id == block.id
                }

        checks.add(
            "Регистрация: " +
                if (installed) "OK"
                else "ОШИБКА"
        )


        val result =
            idCheck &&
            nameCheck &&
            versionCheck &&
            authorCheck &&
            typeCheck &&
            statusCheck &&
            descriptionCheck &&
            capabilitiesCheck &&
            capabilitiesValidCheck &&
            installed


        val report =
            buildString {


                appendLine(
                    if (result)
                        "✅ BLOCK PASSED"
                    else
                        "❌ BLOCK FAILED"
                )


                appendLine()


                checks.forEach {

                    appendLine(it)

                }


                appendLine()


                appendLine(
                    "Найдено возможностей: ${block.capabilities.size}"
                )

            }


        return BlockTestResult(
            passed = result,
            report = report
        )

    }

}
