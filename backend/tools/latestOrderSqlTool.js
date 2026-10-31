import {registerTool} from "./toolRegistry.js";
export function latestOrderSqlTool({explain=true}={}){
 const sql="WITH ranked AS (\n"+
 "  SELECT id, user_id, created_at, amount,\n"+
 "         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC, id DESC) AS rn\n"+
 "  FROM orders\n"+
 ")\nSELECT id, user_id, created_at, amount FROM ranked WHERE rn = 1;";
 const explanation="ROW_NUMBER сортирует заказы по created_at DESC, затем id DESC отдельно для каждого user_id; rn=1 оставляет последний заказ.";
 return {success:true,data:{sql},text:String.fromCharCode(96).repeat(3)+"sql\n"+sql+"\n"+String.fromCharCode(96).repeat(3)+(explain?"\n\n"+explanation:"")};
}
registerTool({name:"latest_order_sql",description:"Verified latest order SQL query by user with tie breaker",arguments:{explain:"Explain"},execute:latestOrderSqlTool});
