// Match fully specified two-step percentage changes to a single task.
export function parseSequencePriceTask(task) {
 const text=String(task||"").replace(/\s+/g," ").trim();
 const price=text.match(/(?:цена|стоимость)\s+(?:товара\s+)?(\d+(?:[.,]\d{1,2})?)\s*(?:₽|руб)/iu);
 const up=text.match(/(?:повысили|увеличили|подняли)\s+на\s+(\d{1,3})\s*%/iu);
 const down=text.match(/(?:затем|потом|после\s+этого)\s+(?:снизили|уменьшили)\s+на\s+(\d{1,3})\s*%/iu);
 if(!price||!up||!down||price.index>up.index||up.index>down.index)return null;
 const other=text.replace(price[0],"").replace(up[0],"").replace(down[0],"");
 if(/\d/.test(other))return null;
 return {price:price[1].replace(",","."),increase:Number(up[1]),decrease:Number(down[1])};
}
