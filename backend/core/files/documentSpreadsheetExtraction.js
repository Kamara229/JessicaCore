import ExcelJS from "exceljs";

const MAX_CELLS=4000;
const MAX_TEXT_CHARS=256000;

/** Never return a deceptively truncated XLSX as a complete document. */
export class SpreadsheetExtractionLimitError extends Error {
    constructor(reason){
        super("Превышен предел полного извлечения XLSX: "+reason+
            ". Раздели таблицу на меньшие файлы.");
        this.name="SpreadsheetExtractionLimitError";
        this.code="DOCUMENT_EXTRACTION_INCOMPLETE";
    }
}

function asCellText(value){
    if(value===null||value===undefined)return "";
    if(typeof value!=="object")return String(value);
    if(Array.isArray(value.richText))
        return value.richText.map(piece=>String(piece?.text||"")).join("");
    if(value.text!==undefined&&value.text!==null)return String(value.text);
    if(value.result!==undefined&&value.result!==null)return String(value.result);
    const valueText=JSON.stringify(value);
    if(typeof valueText!=="string")throw new SpreadsheetExtractionLimitError("нечитаемая ячейка");
    return valueText;
}

/**
 * Extract every populated row of each worksheet or fail atomically.
 * The old implementation silently ignored cells after 4,000 and clipped
 * complex cell representations at 120 characters.
 */
export async function extractCompleteSpreadsheetText(buffer){
    const workbook=new ExcelJS.Workbook();
    await workbook.xlsx.load(buffer);
    const lines=[];
    let cells=0, characters=0;
    const append=value=>{
        characters+=value.length+(lines.length?1:0);
        if(characters>MAX_TEXT_CHARS)
            throw new SpreadsheetExtractionLimitError("объём текста более 256000 символов");
        lines.push(value);
    };
    workbook.eachSheet(sheet=>{
        append("Лист: "+sheet.name);
        sheet.eachRow(row=>{
            const values=row.values.slice(1);
            if(cells+values.length>MAX_CELLS)
                throw new SpreadsheetExtractionLimitError("более 4000 ячеек");
            const text=values.map(asCellText).join(" | ");
            cells+=values.length;
            append(text);
        });
    });
    return lines.join("\n");
}
