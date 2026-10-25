export function dateDiffMode(task) {
 const t=String(task||'').toLowerCase();
 return ['прошло','исключ','elapsed','since','exclud','omitting'].some(x=>t.includes(x))?'elapsed':'difference';
}
