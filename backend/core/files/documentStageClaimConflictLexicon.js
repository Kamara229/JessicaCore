/** Closed lexical pairs for generating review candidates; NOT semantic equivalence. */
export const LEGAL_CLAIM_FORMS=[
    {id:"entry-into-force",
        pattern:/(не\s+)?(вступил\s+в\s+силу)/iu},
    {id:"permission-verb",
        pattern:/(не\s+)?(разрешается|допускается)/iu},
    {id:"modal-capacity",
        pattern:/(не\s+)?(может\s+действовать)/iu},
    {id:"mandatory-adjective",
        pattern:/(не\s*)?(обязателен|обязательна|обязательно)/iu},
    {id:"right-to-act",
        pattern:/(не\s+)?(вправе|имеет\s+право|имеет\s+права|обладает\s+правом)/iu},
    {id:"permissible-adjective",
        pattern:/(не\s*)?(допустимо|допустима|допустим)/iu},
    {id:"consent-required",
        pattern:/(не\s+)?(требуется)/iu}
];

export function canonicalLegalClaimForm(id,form,negative){
    if(id==="right-to-act"){
        // "имеет права" can denote a different plural sense; allow only "не имеет права".
        if(form==="имеет права"&&!negative)return null;
        return "вправе";
    }
    if(id==="permission-verb")return "допускается";
    return form;
}
