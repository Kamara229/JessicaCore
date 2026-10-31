/** Explain the exact shortest path and its exhaustive finite-state certificate. */
export function formatBridgeCrossingProof(times,solved,explain=true) {
  if(!solved)return "";
  const labels=times.map((t,i)=>"№"+(i+1)+" ("+t+" мин)");
  const steps=solved.route.map((s,i)=>
    (i+1)+". "+(s.fromSide===0?"Переходят ":"Возвращаются ")+
    s.people.map(j=>labels[j]).join(" и ")+
    (s.fromSide===0?" → ":" ← ")+"— "+s.cost+" мин.");
  const sum=solved.route.map(s=>s.cost).join(" + ");
  const base="Минимальное время — "+solved.minutes+" мин.\n"+
    steps.join("\n")+"\nСумма: "+sum+" = "+solved.minutes+" мин.";
  if(!explain)return base;
  const masks=Array.from({length:1<<times.length},(_,mask)=>{
    const onRight=times.map((_,i)=>i).filter(i=>mask&(1<<i));
    const name=onRight.length?onRight.map(i=>String(i+1)).join(","):"∅";
    const fmt=s=>s===null?"—":String(s);
    return name+" | "+fmt(solved.distances[mask*2])+" | "+fmt(solved.distances[mask*2+1]);
  });
  return base+"\nДоказательство минимальности: состояние — набор номеров на дальнем берегу и сторона фонаря. "+
    "Из каждого состояния перебраны все законные переходы одного или двух человек; "+
    "стоимость каждого перехода равна времени самого медленного участника. "+
    "При положительных временах алгоритм Дейкстры находит минимальную стоимость каждого достижимого состояния. "+
    "Всего достижимых состояний: "+solved.reachable+". "+
    "Полная таблица кратчайших стоимостей от старта (лев./прав. — положение фонаря):\n"+
    "На дальнем берегу | Лев. | Прав.\n"+masks.join("\n")+
    "\nДля конечного состояния ("+times.map((_,i)=>i+1).join(",")+"; фонарь справа) кратчайшая стоимость — "+
    solved.minutes+" мин. Все допустимые переходы учтены, поэтому более быстрый план невозможен.";
}
