/* Uses the existing result, history and workspace flow for browser Handpick runs. */
function getHandpickOptions() {
  return {
    minWeight: Number(document.getElementById('handpick-min-weight').value),
    avoidBadPairs: document.getElementById('handpick-avoid-pairs').checked,
    minPairScore: Number(document.getElementById('handpick-min-pair').value),
  };
}

function updateHandpickPanel(strategy) {
  const selected = strategy === 'cuo_handpick';
  const supported = selectedOrganism === 'Chlamydomonas reinhardtii chloroplast';
  const pill = document.getElementById('pill-handpick');
  const input = pill.querySelector('input');
  input.disabled = !supported;
  pill.classList.toggle('disabled', !supported);
  pill.title = supported ? '' : (currentLang === 'ja'
    ? 'Handpickはクラミドモナス葉緑体用の参照表です。'
    : 'Handpick uses a Chlamydomonas chloroplast reference.');
  document.getElementById('handpick-panel').hidden = !selected;
  if (selected) {
    const mode = document.getElementById('generation-mode');
    mode.disabled = false;
    document.getElementById('optimization-seed').disabled = mode.value === 'greedy';
    document.getElementById('handpick-min-weight').disabled = mode.value === 'greedy';
    document.getElementById('handpick-min-pair').disabled = !getHandpickOptions().avoidBadPairs;
  }
}

async function runHandpickOptimization(source, protein, settings) {
  if (settings.organism !== 'Chlamydomonas reinhardtii chloroplast') {
    showError(currentLang === 'ja'
      ? 'CUO Handpickはクラミドモナス葉緑体用です。対象生物種を変更してください。'
      : 'CUO Handpick is for the C. reinhardtii chloroplast. Select that organism.');
    return;
  }
  const run = beginLegacyRun('optimize', source, settings);
  const button = document.getElementById('run-btn');
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span> ' + t('btn_optimizing');
  try {
    // Yield so the running state appears before calculating a long protein.
    await new Promise(resolve => setTimeout(resolve, 0));
    const handpick = CUOHandpick.optimize(protein, {
      ...settings.handpick,
      generationMode: settings.generationMode,
      seed: settings.seed,
      originalDNA: settings.inputType === 'dna' ? source : undefined,
    });
    const dna = handpick.sequence;
    const metrics = { gc_percent: calcGC(dna), gc3_percent: calcGC3(dna) };
    const result = {
      predicted_dna: dna,
      sequence: dna,
      runId: run.runId,
      sourceRevisionId: run.revisionId,
      outputRevisionId: crypto.randomUUID(),
      modelVersion: handpick.modelVersion,
      generationMode: handpick.generationMode,
      seed: handpick.seed,
      temperature: null,
      gc3_percent: metrics.gc3_percent,
      warnings: handpick.warnings,
      handpick,
      metrics,
    };
    finishLegacyRun(run, result, {
      organism: settings.organism,
      strategy: 'cuo_handpick',
      proteinInput: handpick.protein + '_',
      originalInput: source,
      originalInputType: settings.inputType,
      dnaOutput: dna,
      gcPercent: metrics.gc_percent,
      csi: null,
      lengthNt: dna.length,
      lengthAa: handpick.protein.length,
      startCodon: dna.slice(0, 3),
      stopCodon: dna.slice(-3),
      seed: handpick.seed,
      warnings: handpick.warnings,
      numCandidates: 1,
    }, () => {
      lastDNA = dna;
      lastProtein = handpick.protein;
      showResults(dna, handpick.protein, 'cuo_handpick', result, settings.organism);
    });
    if (apiAvailable) {
      // Evaluation uses the app's common reference; Handpick CAI stays separately labelled.
      void fetchExtendedMetrics(dna, settings.organism, run).then(evaluation => {
        if (evaluation) {
          result.metrics = evaluation;
          if (legacyCanRender(run) && lastRun?.runId === run.runId) lastRun.metrics = evaluation;
        }
      });
    }
  } catch (error) {
    failLegacyRun(run, error);
    if (legacyCanRender(run)) showError(error.message || String(error));
  } finally {
    resetButton();
  }
}

for (const id of ['handpick-min-weight', 'handpick-avoid-pairs', 'handpick-min-pair']) {
  document.getElementById(id).addEventListener('change', () => {
    updateHandpickPanel(currentStrategy());
    clearLegacyResults('optimize');
  });
}

updateHandpickPanel(currentStrategy());
