/*
 * =========================================================
 * JESSICA TOOL REGISTRY
 * =========================================================
 *
 * Tool Registry — единая точка регистрации
 * всех инструментов Jessica.
 *
 * Planner видит только описание инструментов.
 * TaskRunner вызывает их через executeTool().
 */


/*
 * Зарегистрированные инструменты.
 *
 * Каждый инструмент содержит:
 *
 * name
 * description
 * arguments
 * execute
 */
const tools =
    new Map();


/*
 * =========================================================
 * REGISTER TOOL
 * =========================================================
 */


export function registerTool(
    tool
) {

    if (
        !tool ||
        typeof tool !== "object"
    ) {

        throw new Error(
            "Tool должен быть объектом"
        );

    }


    if (
        typeof tool.name !== "string" ||
        !tool.name.trim()
    ) {

        throw new Error(
            "Tool должен иметь name"
        );

    }


    if (
        typeof tool.description !== "string"
    ) {

        throw new Error(
            `Tool ${tool.name} должен иметь description`
        );

    }


    if (
        typeof tool.execute !== "function"
    ) {

        throw new Error(
            `Tool ${tool.name} должен иметь execute()`
        );

    }


    tools.set(
        tool.name,
        {
            name:
                tool.name,

            description:
                tool.description,

            arguments:
                tool.arguments || {},

            execute:
                tool.execute
        }
    );

}


/*
 * =========================================================
 * GET TOOL
 * =========================================================
 */


export function getTool(
    name
) {

    return tools.get(
        name
    ) || null;

}


/*
 * =========================================================
 * LIST TOOLS
 * =========================================================
 *
 * Возвращаем описание инструментов
 * БЕЗ execute-функций.
 *
 * Именно этот список позже будет
 * передаваться Planner.
 */


export function listTools() {

    return Array
        .from(
            tools.values()
        )
        .map(
            tool => ({
                name:
                    tool.name,

                description:
                    tool.description,

                arguments:
                    tool.arguments
            })
        );

}


/*
 * =========================================================
 * HAS TOOL
 * =========================================================
 */


export function hasTool(
    name
) {

    return tools.has(
        name
    );

}


/*
 * =========================================================
 * EXECUTE TOOL
 * =========================================================
 */


export async function executeTool(
    name,
    args = {}
) {

    const tool =
        getTool(
            name
        );


    if (!tool) {

        return {
            success: false,

            tool:
                name,

            text:
                `Инструмент ${name} не найден`
        };

    }


    try {

        const result =
            await tool.execute(
                args
            );


        return {

            success:
                result?.success !== false,

            tool:
                name,

            ...result

        };


    } catch (error) {

        console.error(
            `Tool execution error [${name}]:`,
            error
        );


        return {
            success: false,

            tool:
                name,

            text:
                `Ошибка выполнения инструмента ${name}`
        };

    }

}


/*
 * =========================================================
 * TOOL COUNT
 * =========================================================
 */


export function getToolCount() {

    return tools.size;

}
