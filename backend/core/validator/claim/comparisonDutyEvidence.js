export function unsupportedComparisonDuty(statement, excerpt) {
  if (!/требу|обязан|должен|необходимо|нужно|следует|must|shall|required/i.test(String(statement || ''))) return false;
  return !/must|shall|require|retain|include|provide|give|condition|обязан|должен/i.test(String(excerpt || ''));
}
