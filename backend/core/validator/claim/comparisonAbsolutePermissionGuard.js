/* Unsupported absolute terms in a legal grant. A permission is not proof of unrestricted rights. */
export function unsupportedAbsolutePermission(cell,quotes) {
  const value=String(cell?.value||"").toLowerCase();
  if(!/без ограничений|неограниченн|unrestricted|without restrictions/i.test(value))return false;
  const proof=(quotes||[]).join(" ").toLowerCase();
  return !/unrestricted|without restrictions|no restrictions/i.test(proof);
}
