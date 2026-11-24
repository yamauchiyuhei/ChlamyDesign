/* Pure, explicit range translation. DNA_TO_AA is defined by index.html. */
function proteinTranslationError(code, message) {
  const error = new Error(message);
  error.code = code;
  return error;
}

function normalizeDnaForTranslation(raw) {
  if (raw == null || raw === '') throw proteinTranslationError('EMPTY', 'Enter a DNA sequence.');
  if (typeof raw !== 'string') throw proteinTranslationError('INVALID_DNA', 'DNA must be text.');
  const lines = raw.replace(/^\uFEFF/, '').split(/\r?\n/).map(line => line.trim()).filter(Boolean);
  const headers = lines.filter(line => line.startsWith('>'));
  if (headers.length > 1) throw proteinTranslationError('MULTIPLE_FASTA', 'Select one FASTA record for translation.');
  if (headers.length && !lines[0].startsWith('>')) {
    throw proteinTranslationError('INVALID_DNA', 'A FASTA header must precede the DNA sequence.');
  }
  const sequence = (headers.length ? lines.slice(1) : lines).join('').replace(/\s+/g, '').toUpperCase();
  if (!sequence) throw proteinTranslationError('EMPTY', 'Enter a DNA sequence.');
  if (sequence.length > 150000) throw proteinTranslationError('TOO_LONG', 'DNA must contain at most 150,000 bases.');
  if (!/^[ACGT]+$/.test(sequence)) throw proteinTranslationError('INVALID_DNA', 'Use only A, C, G and T for translation.');
  return sequence;
}

/** Coordinates refer to the full input DNA, are 1-based and include both ends. */
function translateDnaRange({sequence: raw, start = 1, end, strand = 1} = {}) {
  const sequence = normalizeDnaForTranslation(raw);
  const last = end === undefined ? sequence.length : end;
  if (!Number.isInteger(start) || !Number.isInteger(last) || start < 1 || last < start || last > sequence.length || ![1, -1].includes(strand)) {
    throw proteinTranslationError('INVALID_RANGE', 'Choose an inclusive DNA range within the sequence and a forward or reverse strand.');
  }
  let selectedDna = sequence.slice(start - 1, last);
  if (selectedDna.length % 3 !== 0) {
    throw proteinTranslationError('PARTIAL_CODON', 'The selected range must contain complete codons.');
  }
  if (strand === -1) {
    const complement = {A: 'T', T: 'A', C: 'G', G: 'C'};
    selectedDna = [...selectedDna].reverse().map(base => complement[base]).join('');
  }
  let protein = '';
  let terminalStop = false;
  for (let index = 0; index < selectedDna.length; index += 3) {
    const aminoAcid = DNA_TO_AA[selectedDna.slice(index, index + 3)];
    if (aminoAcid === '*') {
      if (index !== selectedDna.length - 3) {
        throw proteinTranslationError('INTERNAL_STOP', 'The selected reading frame contains an internal stop codon.');
      }
      terminalStop = true;
    } else {
      protein += aminoAcid;
    }
  }
  if (!protein) throw proteinTranslationError('EMPTY_PROTEIN', 'The selected range contains no amino acids.');
  return {sequence, selectedDna, protein, start, end: last, strand, terminalStop, startsWithATG: selectedDna.startsWith('ATG'), geneticCode: 1};
}
