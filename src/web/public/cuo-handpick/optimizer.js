/* Handpick reference-table optimization. Algorithm implementation by ChlamyDesign. */
(() => {
  'use strict';

  const reference = globalThis.CUO_HANDPICK_REFERENCE;
  if (!reference || reference.codons.length !== 64 || reference.pairs.length !== 4096) {
    throw new Error('The CUO Handpick reference tables could not be loaded.');
  }

  const groups = new Map();
  const coordinates = new Map();
  reference.codons.forEach((codon, index) => {
    const aminoAcid = reference.aminoAcids[index];
    if (!groups.has(aminoAcid)) groups.set(aminoAcid, []);
    groups.get(aminoAcid).push(index);
    coordinates.set(codon, index);
  });

  function fraction(value, name) {
    if (!Number.isFinite(value) || value < 0 || value > 1) {
      throw new Error(`${name} must be between 0 and 1.`);
    }
    return value;
  }

  function seededRandom(seed) {
    let state = seed >>> 0;
    return () => {
      state += 0x6d2b79f5;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function normalizeProtein(input) {
    const protein = String(input).toUpperCase().replace(/\s+/g, '').replace(/[_*]$/, '');
    if (!protein || protein.length > 3000 || !/^[ACDEFGHIKLMNPQRSTVWY]+$/.test(protein)) {
      throw new Error('Enter 1–3000 standard amino acids with an optional terminal stop.');
    }
    return protein;
  }

  function translate(dna) {
    if (!/^[ACGT]+$/.test(dna) || dna.length % 3 !== 0) {
      throw new Error('DNA must contain complete A/C/G/T codons.');
    }
    let protein = '';
    for (let offset = 0; offset < dna.length; offset += 3) {
      const aminoAcid = reference.aminoAcids[coordinates.get(dna.slice(offset, offset + 3))];
      if (aminoAcid === '*') {
        if (offset !== dna.length - 3) throw new Error('DNA contains an internal stop codon.');
      } else {
        protein += aminoAcid;
      }
    }
    return protein;
  }

  function selectCodon(aminoAcid, mode, minWeight, random, originalCodon) {
    const group = groups.get(aminoAcid);
    if (mode === 'sample') {
      const eligible = group.filter(index => reference.weights[index] >= minWeight);
      const total = eligible.reduce((sum, index) => sum + reference.weights[index], 0);
      if (total <= 0) throw new Error(`No Handpick codon satisfies the threshold for ${aminoAcid}.`);
      let chance = random() * total;
      for (const index of eligible) {
        chance -= reference.weights[index];
        if (chance < 0) return index;
      }
      return eligible[eligible.length - 1];
    }

    // CUO keeps the input codon when several synonyms share the highest count.
    const originalIndex = coordinates.get(originalCodon);
    let best = group.includes(originalIndex) ? originalIndex : group[0];
    for (const index of group) {
      if (reference.frequencies[index] > reference.frequencies[best]) best = index;
    }
    return best;
  }

  function repairPairs(selected, threshold) {
    // Scan left to right, keeping the first codon fixed while choosing its successor.
    for (let position = 0; position + 1 < selected.length; position++) {
      const first = selected[position];
      const second = selected[position + 1];
      if (reference.pairs[first * 64 + second] > threshold) continue;
      let bestScore = threshold;
      let best = second;
      for (const index of groups.get(reference.aminoAcids[second])) {
        const score = reference.pairs[first * 64 + index];
        if (score > bestScore) {
          bestScore = score;
          best = index;
        }
      }
      selected[position + 1] = best;
    }
  }

  function optimize(input, options = {}) {
    const protein = normalizeProtein(input);
    const mode = options.generationMode ?? 'greedy';
    const seed = options.seed ?? 0;
    const minWeight = fraction(options.minWeight ?? 0.5, 'Minimum codon weight');
    const minPairScore = fraction(options.minPairScore ?? 0.5, 'Minimum pair score');
    if (!['greedy', 'sample'].includes(mode)) throw new Error('Unknown Handpick generation mode.');
    if (!Number.isInteger(seed) || seed < 0 || seed > 2147483640) {
      throw new Error('Seed must be an integer between 0 and 2147483640.');
    }
    const originalDNA = options.originalDNA?.toUpperCase().replace(/\s+/g, '');
    if (originalDNA && translate(originalDNA) !== protein) {
      throw new Error('The input DNA does not encode the supplied protein.');
    }

    const random = seededRandom(seed);
    const selected = [...protein, '*'].map((aminoAcid, position) => selectCodon(
      aminoAcid, mode, minWeight, random, originalDNA?.slice(position * 3, position * 3 + 3),
    ));
    if (options.avoidBadPairs) repairPairs(selected, minPairScore);
    const dna = selected.map(index => reference.codons[index]).join('');
    if (translate(dna) !== protein) throw new Error('Handpick output failed protein validation.');

    const weights = selected.slice(0, -1).filter(index => groups.get(reference.aminoAcids[index]).length > 1)
      .map(index => reference.weights[index]);
    let cai = 1;
    if (weights.length > 0) {
      cai = weights.some(value => value === 0) ? 0
        : Math.exp(weights.reduce((sum, value) => sum + Math.log(value), 0) / weights.length);
    }
    const pairScores = selected.slice(0, -1).map((first, index) => reference.pairs[first * 64 + selected[index + 1]]);
    const unresolvedPairs = pairScores.filter(score => score <= minPairScore).length;
    const warnings = [];
    if (options.avoidBadPairs && unresolvedPairs) {
      warnings.push(`${unresolvedPairs} codon pair(s) remain at or below the requested Handpick score.`);
    }

    return {
      sequence: dna,
      protein,
      modelVersion: reference.version,
      referenceName: reference.name,
      referenceSource: reference.source,
      referenceHashes: { ...reference.sha256 },
      generationMode: mode,
      seed: mode === 'sample' ? seed : null,
      minWeight,
      avoidBadPairs: Boolean(options.avoidBadPairs),
      minPairScore,
      handpickCAI: cai,
      unresolvedPairs,
      warnings,
    };
  }

  globalThis.CUOHandpick = Object.freeze({ optimize, translate, version: reference.version });
})();
