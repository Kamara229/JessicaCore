/**
 * Canonical document URLs only. These are discovery references, not
 * verified legal conclusions; every plan still retrieves actual text.
 */
export const KNOWN_LICENSE_SOURCES = [
    ["Apache-2.0", /\bapache(?:\s+license)?\s*(?:2(?:\.0)?|v2)\b/i, "https://www.apache.org/licenses/LICENSE-2.0.txt"],
    ["GPL-3.0", /(?:\bgpl\s*(?:v\s*)?3(?:\.0)?\b|\bgnu\s+general\s+public\s+license\s*(?:v\s*)?3(?:\.0)?\b)/i, "https://www.gnu.org/licenses/gpl-3.0.txt"],
    ["MPL-2.0", /(?:mpl|mozilla public license)\s*2(?:\.0)?/i, "https://www.mozilla.org/media/MPL/2.0/index.txt"],
    ["BSD-3-Clause", /\bbsd[\s-]*3[\s-]*clause\b/i, "https://opensource.org/license/bsd-3-clause"],
    ["BSD-2-Clause", /\bbsd[\s-]*2[\s-]*clause\b/i, "https://opensource.org/license/bsd-2-clause"],
    ["MIT", /\bmit(?:\s+license)?\b/i, "https://opensource.org/license/mit"]
];
