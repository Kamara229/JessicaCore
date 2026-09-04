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


        val checks = mutableListOf<String>()


        val nameCheck =
            block.name.isNotBlank()

        checks.add(
            "Имя блока: " +
                    if(nameCheck) "OK"
                    else "ОШИБКА"
        )


        val versionCheck =
            block.version.isNotBlank()

        checks.add(
            "Версия: " +
                    if(versionCheck) "OK"
                    else "ОШИБКА"
        )


        val statusCheck =
            block.status == "ACTIVE"

        checks.add(
            "Статус: " +
                    if(statusCheck) "OK"
                    else "ОШИБКА"
        )


        val installed =
            manager.getBlocks()
                .contains(block)

        checks.add(
            "Регистрация: " +
                    if(installed) "OK"
                    else "ОШИБКА"
        )


        val result =
            nameCheck &&
            versionCheck &&
            statusCheck &&
            installed


        val report =
            buildString {

                appendLine(
                    if(result)
                        "✅ BLOCK PASSED"
                    else
                        "❌ BLOCK FAILED"
                )

                appendLine()

                checks.forEach {
                    appendLine(it)
                }

            }


        return BlockTestResult(
            passed = result,
            report = report
        )

    }

}
