import express from "express";
import { createRequire } from "node:module";
import mammoth from "mammoth";

import { requireJessicaAuthorization } from "../auth/jessicaAuth.js";
import { pageDocumentText, parseDocumentTextOffset } from "../../core/files/documentTextWindow.js";
import { buildDocumentSectionIndex } from "../../core/files/documentSectionIndex.js";
import {parseSectionIndex,selectDocumentSectionPage} from "../../core/files/documentSectionWindow.js";
import {buildDocumentPageBatch} from "../../core/files/documentPageBatch.js";
import {prepareExtractedDocumentText} from "../../core/files/documentExtractedTextIntegrity.js";
import {extractCompleteSpreadsheetText,SpreadsheetExtractionLimitError} from "../../core/files/documentSpreadsheetExtraction.js";

const require = createRequire(import.meta.url);
const parsePdf = require("pdf-parse");
const MAX_BYTES = 5 * 1024 * 1024;

const supported = new Set(["pdf", "docx", "txt", "md", "csv", "json", "xlsx", "py", "pyw", "js", "jsx", "mjs", "cjs", "ts", "tsx", "mts", "cts", "java", "kt", "kts", "swift", "go", "rs", "c", "h", "cpp", "hpp", "cc", "cxx", "cs", "fs", "fsx", "m", "mm", "php", "rb", "dart", "scala", "sc", "hs", "erl", "ex", "exs", "clj", "cljs", "lua", "pl", "pm", "r", "jl", "sql", "sh", "bash", "ps1", "psm1", "fish", "bat", "cmd", "groovy", "gradle", "sol", "move", "cob", "cbl", "f", "for", "f90", "f95", "pas", "pp", "dpr", "asm", "s", "zig", "nim", "ml", "mli", "vb", "bas", "cls", "frm", "tf", "tfvars", "hcl", "yaml", "yml", "xml", "xsd", "xsl", "toml", "graphql", "gql", "proto", "dockerfile", "mk", "makefile", "cu", "cuh", "wat", "awk", "lisp", "lsp", "scm", "vue", "svelte", "ipynb", "html", "htm", "css", "scss", "sass", "less", "jsonl"]);

function cleanName(value) {
    const valueString = String(value || "");
    const fileName = valueString.split(/[\\/]/).pop() || "file";
    return fileName.replace(/[\r\n\t<>]/g, " ").slice(0, 100);
}

function getExtension(name) {
    return name.split(".").pop().toLowerCase();
}

function ensureSignature(ext, buffer) {
    if (ext === "pdf" && buffer.subarray(0, 5).toString("ascii") !== "%PDF-") {
        throw new Error("Файл не является PDF");
    }
    if (["docx", "xlsx"].includes(ext) &&
        !(buffer[0] === 0x50 && buffer[1] === 0x4b)) {
        throw new Error("Некорректный формат Office");
    }
}

async function extractText(ext, buffer) {
    if (ext === "pdf") return (await parsePdf(buffer)).text || "";
    if (ext === "docx") return (await mammoth.extractRawText({ buffer })).value || "";
    if (ext === "xlsx") return extractCompleteSpreadsheetText(buffer);
    // Avoid executing JSON, markup, or scripts from uploaded text files.
    return new TextDecoder("utf-8", { fatal: true }).decode(buffer);
}

export function registerDocumentExtractRoute(app) {
    app.post(
        "/api/files/extract",
        requireJessicaAuthorization,
        express.raw({ type: "application/octet-stream", limit: "5mb" }),
        async (req, res) => {
            try {
                const encoded = String(req.get("X-Jessica-Filename") || "");
                const name = cleanName(decodeURIComponent(encoded));
                const ext = getExtension(name);
                if (!supported.has(ext)) {
                    return res.status(415).json({
                        success: false,
                        text: "Поддерживаются PDF, DOCX, XLSX и текстовые файлы исходного кода"
                    });
                }
                const buffer = req.body;
                if (!Buffer.isBuffer(buffer) || !buffer.length || buffer.length > MAX_BYTES) {
                    return res.status(413).json({success: false, text: "Размер файла должен быть от 1 байта до 5 МБ"});
                }
                ensureSignature(ext, buffer);
                const offset = parseDocumentTextOffset(req.get("X-Jessica-Text-Offset"));
                if (offset === null) {
                    return res.status(400).json({success: false, text: "Некорректное смещение текста документа"});
                }
                const extraction = prepareExtractedDocumentText(
                    await extractText(ext, buffer)
                );
                const complete = extraction.text;
                if (!extraction.hasReadableContent) {
                    return res.status(422).json({
                        success: false,
                        text: "В файле нет доступного текста. Сканированные PDF пока не поддерживаются."
                    });
                }
                const sectionIndex = buildDocumentSectionIndex(complete);
                const requested = req.get("X-Jessica-Section-Index");
                const selected = requested === undefined ? null : parseSectionIndex(requested);
                if (requested !== undefined && selected === null) {
                    return res.status(400).json({success:false, text:"Некорректный номер раздела"});
                }
                const page = selected === null ? pageDocumentText(complete, offset)
                    : selectDocumentSectionPage(complete, sectionIndex, selected,
                        req.get("X-Jessica-Text-Offset") === undefined ? null : offset);
                if (!page) {
                    return res.status(416).json({success:false, text:"Смещение или раздел за пределами документа"});
                }
                console.log("Jessica document parsed:", {
                    extension: ext,
                    bytes: buffer.length,
                    extractedChars: complete.length,
                    nullCharactersRemoved: extraction.nullCharactersRemoved,
                    truncated: page.truncated,
                    offset: page.offset,
                    hasMore: page.hasMore
                });
                const batchMode=req.get("X-Jessica-Pages");
                if(batchMode!==undefined && batchMode!=="all")
                    return res.status(400).json({success:false,text:"Неизвестный режим извлечения"});
                if(batchMode==="all"){
                    if(selected!==null||req.get("X-Jessica-Text-Offset")!==undefined)
                        return res.status(400).json({success:false,text:"Нельзя совмещать страницы и смещение"});
                    const pages=buildDocumentPageBatch(complete,name);
                    if(!pages)
                        return res.status(413).json({success:false,text:
                            "Документ превышает лимит единого набора страниц; требуется обработка по частям."});
                    const payload={success:true,name,...page,pages,sectionIndex};
                    if(Buffer.byteLength(JSON.stringify(payload),"utf8")>850000)
                        return res.status(413).json({success:false,text:
                            "Извлечённый текст слишком большой для одного запроса."});
                    return res.json(payload);
                }
                return res.json({success: true, name, ...page,
                    sectionIndex});
            } catch (error) {
                if(error instanceof SpreadsheetExtractionLimitError)
                    return res.status(413).json({
                        success:false,stage:"document-extraction-incomplete",
                        text:error.message
                    });
                console.warn("Jessica document extraction failed:", error?.message || "unknown");
                return res.status(422).json({success: false, text: "Не удалось прочитать файл. Проверь формат и попробуй другой."});
            }
        }
    );
}
