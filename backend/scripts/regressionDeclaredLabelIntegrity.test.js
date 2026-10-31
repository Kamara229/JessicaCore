import test from 'node:test';
import assert from 'node:assert/strict';
import {undeclaredQuotedLabel} from '../core/validator/declaredLabelIntegrity.js';
const task='Коробки: яблоки, груши, смесь. Все ярлыки неверны.';
test('user-defined label substitution is rejected',()=>{
 assert.equal(undeclaredQuotedLabel(task,'В коробке «яблоки и апельсины»')?.valid,false);
 assert.equal(undeclaredQuotedLabel(task,'В коробке «смесь»'),null);
});
