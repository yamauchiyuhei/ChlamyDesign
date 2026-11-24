import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import vm from 'node:vm';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), 'web/public/cuo-handpick');
const context = vm.createContext({});
vm.runInContext(readFileSync(path.join(root, 'reference.js'), 'utf8'), context);
vm.runInContext(readFileSync(path.join(root, 'optimizer.js'), 'utf8'), context);
const optimizer = context.CUOHandpick;
const reference = context.CUO_HANDPICK_REFERENCE;

// An independent standard-code decoder catches incorrect reference coordinate order.
const codons = ['TTT TTC TTA TTG', 'TCT TCC TCA TCG', 'TAT TAC TAA TAG', 'TGT TGC TGA TGG',
  'CTT CTC CTA CTG', 'CCT CCC CCA CCG', 'CAT CAC CAA CAG', 'CGT CGC CGA CGG',
  'ATT ATC ATA ATG', 'ACT ACC ACA ACG', 'AAT AAC AAA AAG', 'AGT AGC AGA AGG',
  'GTT GTC GTA GTG', 'GCT GCC GCA GCG', 'GAT GAC GAA GAG', 'GGT GGC GGA GGG'].join(' ').split(' ');
const aminoAcids = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG';
const standardCode = Object.fromEntries(codons.map((codon, index) => [codon, aminoAcids[index]]));
function decode(dna) {
  return dna.match(/.../g).map(codon => standardCode[codon]).join('');
}

test('all three published reference files retain the upstream bytes', () => {
  for (const extension of ['wt', 'cft', 'cpt']) {
    const bytes = readFileSync(path.join(root, `C.reinhardtii Handpick.${extension}`));
    assert.equal(createHash('sha256').update(bytes).digest('hex'), reference.sha256[extension]);
  }
  const counts = readFileSync(path.join(root, 'C.reinhardtii Handpick.cft'), 'utf8').trim().split(/\s+/).map(Number);
  assert.deepEqual(Array.from(reference.frequencies), counts.slice(0, 64));
  assert.equal(reference.pairs.length, 4096);
  assert.equal(reference.codons.length, 64);
  for (let index = 0; index < 64; index++) {
    assert.equal(reference.aminoAcids[index], standardCode[reference.codons[index]]);
  }
});

test('CUO Handpick selects TTC for phenylalanine and TAA for the terminal stop', () => {
  assert.equal(optimizer.optimize('MFF_').sequence, 'ATGTTCTTCTAA');
  assert.equal(optimizer.optimize(' m f f * ').sequence, 'ATGTTCTTCTAA');
});

test('both generation modes preserve every standard amino acid and a terminal stop', () => {
  const protein = 'MACDEFGHIKLNPQRSTVWY'.repeat(20);
  for (const generationMode of ['greedy', 'sample']) {
    for (const avoidBadPairs of [false, true]) {
      const result = optimizer.optimize(protein, { generationMode, avoidBadPairs, seed: 17 });
      assert.equal(decode(result.sequence), protein + '*');
      assert.equal(result.sequence.length, (protein.length + 1) * 3);
    }
  }
});

test('sampling follows the original .wt values and is reproducible', () => {
  const protein = 'M' + 'F'.repeat(2000);
  const options = { generationMode: 'sample', minWeight: 0.5, seed: 824 };
  const first = optimizer.optimize(protein, options);
  assert.equal(first.sequence, optimizer.optimize(protein, options).sequence);
  assert.notEqual(first.sequence, optimizer.optimize(protein, { ...options, seed: 825 }).sequence);
  const tttCount = first.sequence.match(/.../g).filter(codon => codon === 'TTT').length;
  assert.ok(tttCount > 650 && tttCount < 850, `TTT count ${tttCount} should reflect 0.6 / 1.6`);
  assert.ok(!optimizer.optimize(protein, { ...options, minWeight: 0.65 }).sequence.includes('TTT'));
});

test('DNA input preserves synonymous ties and protein identity', () => {
  const originalDNA = 'ATGTTTTTCTAG';
  const result = optimizer.optimize('MFF', { originalDNA });
  assert.equal(result.sequence, 'ATGTTCTTCTAA');
  assert.throws(() => optimizer.optimize('MFA', { originalDNA }), /does not encode/);
});

test('invalid proteins, internal stops and invalid settings fail before producing DNA', () => {
  for (const protein of ['', 'MA*FF', 'MAX', 'M'.repeat(3001)]) {
    assert.throws(() => optimizer.optimize(protein));
  }
  assert.throws(() => optimizer.optimize('MFF', { minWeight: NaN }));
  assert.throws(() => optimizer.optimize('MFF', { minPairScore: 1.1 }));
  assert.throws(() => optimizer.optimize('MFF', { seed: -1 }));
  assert.throws(() => optimizer.optimize('MFF', { originalDNA: 'ATGTAATTC' }), /internal stop/);
});
