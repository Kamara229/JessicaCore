import express from "express";
import { createRequire } from "node:module";
import mammoth from "mammoth";
import ExcelJS from "exceljs";
import { requireJessicaAuthorization } from "../auth/jessicaAuth.js";

const require = createRequire(import.meta.url);
const parsePdf = require("pdf-parse");
const MAX_BYTES = 5 * 1024 * 1024;
const MAX_TEXT = 16000;
const supported = new Set(["pdf", "docx", "txt", "md", "csv", "json", "xlsx"]);

function cleanName(value) {
    const valueString = String(value || "");
    const fileName = valueString.split(/[\\/]/).pop() || "file";
    return fileName.replace(/[\\r\\n\\t<>]/g, " ").slice(0, 100);
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

async function extractSpreadsheet(buffer) {
    const workbook = new ExcelJS.Workbook();
    await workbook.xlsx.load(buffer);
    const lines = [];
    let cells = 0;
    workbook.eachSheet(sheet => {
        lines.push("Лист: " + sheet.name);
        sheet.eachRow(row => {
            if (cells >= 4000) return;
            const values = row.values.slice(1).map(v => {
                if (v && typeof v === "object") {
                    if (v.text) return String(v.text);
                    if (v.result != null) return String(v.result);
                    return JSON.stringify(v).slice(0, 120);
                }
                return String(v ?? "");
            });
            cells += values.length;
            lines.push(values.join(" | "));
        });
    });
    return lines.join("\\n");
}

async function extractText(ext, buffer) {
    if (ext === "pdf") return (await parsePdf(buffer)).text || "";
    if (ext === "docx") return (await mammoth.extractRawText({ buffer })).value || "";
    if (ext === "xlsx") return extractSpreadsheet(buffer);
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
                        text: "Поддерживаются PDF, DOCX, XLSX, TXT, MD, CSV, JSON"
                    });
                }
                const buffer = req.body;
                if (!Buffer.isBuffer(buffer) || !buffer.length || buffer.length > MAX_BYTES) {
                    return res.status(413).json({success: false, text: "Размер файла должен быть от 1 байта до 5 МБ"});
                }
                ensureSignature(ext, buffer);
                const complete = (await extractText(ext, buffer)).replace(/\\u0000/g, "").trim();
                if (!complete) {
                    return res.status(422).json({
                        success: false,
                        text: "В файле нет доступного текста. Сканированные PDF пока не поддерживаются."
                    });
                }
                const text = complete.slice(0, MAX_TEXT);
                console.log("Jessica document parsed:", {
                    extension: ext,
                    bytes: buffer.length,
                    extractedChars: complete.length,
                    truncated: complete.length > MAX_TEXT
                });
                return res.json({
                    success: true,
                    name,
                    text,
                    truncated: complete.length > MAX_TEXT,
                    extractedChars: complete.length
                });
            } catch (error) {
                console.warn("Jessica document extraction failed:", error?.message || "unknown");
                return res.status(422).json({success: false, text: "Не удалось прочитать файл. Проверь формат и попробуй другой."});
            }
        }
    );
}
