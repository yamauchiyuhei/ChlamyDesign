/* Shared navigation and explicit sequence handoff; no model runs on tab changes. */
let workspaceMount;
let analysisOrigin = null;
const analysisOrigins = new Map();
let legacyAuthEpoch = 0;
let legacyAuthIdentity = null;
const legacyActiveRuns = new Map();
const legacyCompletedRuns = new Map();
const integrationLabels = {
  ja: {runs:'実行一覧',workspace:'ワークスペース',add:'ワークスペースに追加',back:'ワークスペースへ戻る',origin:'解析対象',edited:'配列を編集したため、ワークスペースとの関連付けを解除しました。'},
  en: {runs:'Runs',workspace:'Workspace',add:'Add to workspace',back:'Back to workspace',origin:'Analysis input',edited:'Sequence edited. This analysis is no longer linked to the workspace.'}
};
function integrationText(key) { return integrationLabels[currentLang === 'ja' ? 'ja' : 'en'][key]; }
function integrationNotice(message) {
  document.querySelector('.integration-notice')?.remove();
  const notice = document.createElement('div');
  notice.className = 'integration-notice'; notice.setAttribute('role','status'); notice.textContent = message;
  document.body.append(notice); setTimeout(() => notice.remove(), 7000);
}
window.updateIntegrationLabels = function () {
  document.querySelectorAll('[data-integration]').forEach(el => { el.textContent = integrationText(el.dataset.integration); });
  renderAnalysisOrigin(); window.renderProteinBridge?.();
};
window.ensureCodonWorkspace = function () {
  if (!workspaceMount) workspaceMount = (async () => {
    if (!window.mountCodonWorkspace) await import('/workspace-assets/workspace.js');
    await window.mountCodonWorkspace(document.getElementById('workspace-root'));
  })().catch(error => {
    workspaceMount = undefined;
    integrationNotice(currentLang === 'ja' ? 'ワークスペースを開けませんでした。ページを再読み込みしてください。' : 'Workspace could not load. Please reload the page.');
    throw error;
  });
  return workspaceMount;
};
async function showCommonRuns(){
  try{window.codonPendingShowRuns=true;closeHistoryPanel();switchTab('workspace');await window.ensureCodonWorkspace();window.dispatchEvent(new CustomEvent('codon:show-runs'));}
  catch(error){integrationNotice(currentLang==='ja'?'実行一覧を開けませんでした。ページを再読み込みしてください。':'Could not open run history. Please reload the page.');}
}
async function importToWorkspace(detail) {
  try {
    closeHistoryPanel(); switchTab('workspace');
    await window.ensureCodonWorkspace();
    window.dispatchEvent(new CustomEvent('codon:workspace-import', {detail:{id:crypto.randomUUID(),...detail}}));
    window.scrollTo({top:0});
  } catch(error) { console.error('Workspace import failed', error); }
}
function sendResultToWorkspace() {
  if (!lastRun?.dna) return;
  const run = {...lastRun};
  return importToWorkspace({name: `${run.strategy} · ${run.protein.length} aa`,sequence:run.dna,kind:'cds',species:run.organism,source:'CodonDesign',protein:run.protein,strategy:run.strategy,seed:run.seed,temperature:run.temperature,warnings:run.warnings,historyId:run.historyId,runId:run.runId,sourceRevisionId:run.sourceRevisionId,modelVersion:run.modelVersion,generationMode:run.generationMode});
}
function sendHistoryToWorkspace(id) {
  const item = historyItems.find(h => h.id === id);
  if (!item?.dnaOutput) return;
  let stored=item.result;try{if(!stored&&item.resultJSON)stored=JSON.parse(item.resultJSON);}catch{/* Older history may contain inputs only. */}
  return importToWorkspace({name:item.customName || `${item.strategy || 'CDS'} · ${item.lengthAa || '?'} aa`,sequence:item.dnaOutput,kind:'cds',species:item.organism,source:'CodonDesign history',protein:item.proteinInput,strategy:item.strategy,seed:item.seed,temperature:item.temperature,warnings:item.warnings,historyId:item.id,runId:item.runId||stored?.runId||item.id,sourceRevisionId:item.sourceRevisionId||item.revisionId||stored?.sourceRevisionId,modelVersion:item.modelVersion||stored?.modelVersion,generationMode:item.generationMode||stored?.generationMode||item.settings?.generationMode});
}
function setInputType(kind) {
  const input = document.querySelector(`input[name="input-type"][value="${kind}"]`);
  if (input) { input.checked = true; input.dispatchEvent(new Event('change', {bubbles:true})); }
}
function legacyPanel(type) { return type === 'optimize' ? 'codon' : ['restriction','rna'].includes(type) ? 'seqanalysis' : 'protein'; }
function normalizedInput(value) { return String(value || '').trim().toUpperCase().replace(/\s+/g,''); }
function legacyInput(type) { return normalizedInput(document.getElementById(type === 'optimize' ? 'seq-input' : ['restriction','rna'].includes(type) ? 'sa-dna-input' : 'pa-protein-input').value); }
function analysisOriginForPanel(panel) {
  const candidate=analysisOrigin?.panel===panel?analysisOrigin:analysisOrigins.get(panel);
  const input=document.getElementById(panel==='codon'?'seq-input':panel==='protein'?'pa-protein-input':'sa-dna-input');
  return candidate&&normalizedInput(candidate.sequence)===normalizedInput(input.value)?candidate:null;
}
function activateAnalysisPanel(panel) {
  if(analysisOrigin)analysisOrigins.set(analysisOrigin.panel,analysisOrigin);
  analysisOrigin=['codon','seqanalysis','protein'].includes(panel)?analysisOriginForPanel(panel):null;
  renderAnalysisOrigin();if(panel==='seqanalysis')window.renderProteinBridge?.();
}
window.activateAnalysisPanel=activateAnalysisPanel;
function legacySettings(type) {
  if (type === 'optimize') return {inputType,organism:selectedOrganism,strategy:currentStrategy(),temperature:Number(document.getElementById('slider-temp').value),seed:Number(document.getElementById('optimization-seed').value),generationMode:document.getElementById('generation-mode').value,execution:currentStrategy()==='cuo_handpick'?'browser':window.codonJobsAvailable&&window.runCodonOptimization?'durable':'synchronous',...(currentStrategy()==='cuo_handpick'?{handpick:getHandpickOptions()}:{} )};
  const origin = analysisOriginForPanel(legacyPanel(type));
  const source = origin ? (['restriction','rna'].includes(type) ? {sourceContext:{sourceKind:origin.sourceKind||null,cdsBlocks:origin.cdsBlocks||[]}} : origin.translation ? {translation:origin.translation} : {}) : {};
  if (type === 'restriction') return {...source,enzymeSet:document.getElementById('re-enzyme-set').value};
  if (type === 'rna') return {...source,region:document.getElementById('rna-region').value,utr5:normalizedInput(document.getElementById('rna-utr-input').value)};
  if (type === 'blast') return {...source,database:document.getElementById('blast-db').value,program:'blastp',hitListSize:10};
  return {...source,provider:'ESM Atlas',format:'pdb'};
}
function legacyAuthChanged() {
  const next = window.currentUser || null;
  if (legacyAuthIdentity !== next) { legacyAuthIdentity = next; legacyAuthEpoch++; clearLegacyResults(); analysisOrigins.clear(); analysisOrigin=null; renderAnalysisOrigin(); window.renderProteinBridge?.(true); }
}
function legacySameSession(run) { return run.authSession === (window.currentUser || null) && run.authEpoch === legacyAuthEpoch; }
function legacyRunEvent(run, status, result, error) {
  if (!legacySameSession(run)) return;
  const {authSession, authEpoch, fingerprint, ...detail} = run;
  window.dispatchEvent(new CustomEvent('codon:analysis-result', {detail:{...detail, status, result:result ?? null, error:error || null, completedAt:status === 'running' ? null : new Date().toISOString()}}));
}
function beginLegacyRun(type, sequence, settings = legacySettings(type)) {
  const kind = type === 'optimize' ? (settings.inputType === 'dna' ? 'dna' : 'protein') : ['restriction','rna'].includes(type) ? 'dna' : 'protein';
  const origin = analysisOriginForPanel(legacyPanel(type));
  const run = {runId:crypto.randomUUID(),type,sequence,kind,settings:JSON.parse(JSON.stringify(settings)),createdAt:new Date().toISOString(),actorKey:window.currentUser ? 'user:' + window.currentUser.uid : 'guest',authSession:window.currentUser || null,authEpoch:legacyAuthEpoch,workspaceId:origin?.workspaceId || null,nodeId:origin?.nodeId || null,revisionId:origin?.revisionId || null,species:origin?.species || (type === 'optimize' ? settings.organism : null),name:origin?.name || (currentLang === 'ja' ? '直接入力' : 'Direct input'),scope:origin?.scope || 'standalone',sourceRunId:origin?.sourceRunId || null,fingerprint:JSON.stringify({sequence,settings,origin:origin ? [origin.workspaceId,origin.nodeId,origin.revisionId] : null})};
  legacyActiveRuns.set(type, run);
  clearLegacyResults(type); renderAnalysisOrigin(); legacyRunEvent(run, 'running');
  return Object.freeze(run);
}
function legacyCanRender(run) {
  if (!legacySameSession(run) || legacyActiveRuns.get(run.type)?.runId !== run.runId) return false;
  const origin = analysisOriginForPanel(legacyPanel(run.type));
  return run.fingerprint === JSON.stringify({sequence:legacyInput(run.type),settings:legacySettings(run.type),origin:origin ? [origin.workspaceId,origin.nodeId,origin.revisionId] : null});
}
function finishLegacyRun(run, result, historyData, render) {
  if (!legacySameSession(run)) return;
  legacyRunEvent(run, 'completed', result);
  if (historyData) saveHistory(run.type, {...historyData,result,settings:run.settings,runId:result?.runId||run.runId,sourceRevisionId:result?.sourceRevisionId||run.revisionId,modelVersion:result?.modelVersion||null,generationMode:result?.generationMode||run.settings.generationMode||null,sourceSequence:run.sequence,inputKind:run.kind,workspaceId:run.workspaceId,nodeId:run.nodeId,revisionId:run.revisionId,analysisScope:run.scope,species:run.species}, run);
  if (legacyCanRender(run)) { render(); legacyCompletedRuns.set(run.type,run); renderAnalysisOrigin(); }
  else integrationNotice(currentLang === 'ja' ? '以前の入力の計算が完了しました。実行履歴から確認できます。' : 'A previous input has finished. Open it from run history.');
}
function failLegacyRun(run, error) { legacyRunEvent(run, 'failed', null, error?.message || String(error)); }
function clearLegacyResults(type) {
  if(type)legacyCompletedRuns.delete(type);else legacyCompletedRuns.clear();
  const ids = type ? ({optimize:['results'],restriction:['re-results'],rna:['rna-results'],esmfold:['esm-results'],blast:['blast-results']}[type] || []) : ['results','re-results','rna-results','esm-results','blast-results'];
  ids.forEach(id => {const el=document.getElementById(id); if(id === 'results')el.classList.remove('visible'); else el.style.display='none';const error=document.getElementById(id==='results'?'err-msg':id.replace('-results','-err-msg'));error?.classList.remove('visible');});
  if (!type || type === 'optimize') {lastDNA='';lastProtein='';lastRun=null;}
  if (!type || type === 'esmfold') {lastPDB='';esmViewer=null;}
  if (!type || type === 'rna') rnaCurrentResults=null;
  if (!type || type === 'blast') blastRID=null;
}
function renderAnalysisOrigin() {
  if(analysisOrigin)analysisOrigins.set(analysisOrigin.panel,analysisOrigin);
  for (const panel of ['codon','seqanalysis','protein']) {
    let banner = document.querySelector(`#panel-${panel} .analysis-origin`);
    if (!banner) {
      banner = document.createElement('div'); banner.className = 'analysis-origin';
      const text = document.createElement('span'); const button = document.createElement('button');
      button.type = 'button'; button.addEventListener('click', () => {switchTab('workspace'); if(analysisOrigin?.workspaceId)window.dispatchEvent(new CustomEvent('codon:focus-workspace',{detail:analysisOrigin}));});
      banner.append(text,button); document.getElementById(`panel-${panel}`).prepend(banner);
    }
    const origin = analysisOriginForPanel(panel);
    const input = document.getElementById(panel==='codon'?'seq-input':panel==='protein'?'pa-protein-input':'sa-dna-input');
    const length = normalizedInput(input.value).length;
    const kind = panel === 'protein' || (panel === 'codon' && inputType === 'protein') ? 'aa' : 'nt';
    banner.hidden = !length;
    banner.querySelector('span').textContent = `${integrationText('origin')}: ${origin?.name || (currentLang === 'ja' ? '直接入力' : 'Direct input')} · ${origin?.scope || (kind === 'aa' ? 'Protein' : 'DNA')} · ${length} ${kind}${origin?.revisionId ? ' · ' + (currentLang === 'ja' ? '配列版 ' : 'Revision ') + origin.revisionId.slice(0,8) : ''}${origin?.historical ? (currentLang === 'ja' ? ' · 保存済みの結果' : ' · Saved result') : ''}`;
    const button=banner.querySelector('button');button.hidden=!origin?.workspaceId;button.textContent=integrationText('back');
    if(panel==='protein'){
      let dnaBack=banner.querySelector('.protein-dna-back');
      if(!dnaBack){dnaBack=document.createElement('button');dnaBack.type='button';dnaBack.className='protein-dna-back';dnaBack.addEventListener('click',backToSequenceAnalysis);banner.append(dnaBack);}
      dnaBack.hidden=!origin?.translation?.dnaSequence;dnaBack.textContent=currentLang==='ja'?'元のDNAに戻る':'Back to source DNA';
    }
  }
}
window.addEventListener('codon:analyze', event => {
  const detail = event.detail;
  if (!detail || !['dna','protein'].includes(detail.kind) || typeof detail.sequence !== 'string') return;
  const protein = detail.kind === 'protein';
  (protein?['esmfold','blast']:['restriction','rna']).forEach(clearLegacyResults);
  analysisOrigin = {...detail,panel:protein ? 'protein' : 'seqanalysis'};
  const input = document.getElementById(protein ? 'pa-protein-input' : 'sa-dna-input');
  input.value = detail.sequence;
  document.getElementById(protein ? 'pa-seq-len' : 'sa-seq-len').textContent = detail.sequence.length + (protein ? ' aa' : ' nt');
  if (!protein) document.getElementById('rna-utr-input').value = '';
  renderAnalysisOrigin(); if(!protein)window.renderProteinBridge?.(true); switchTab(protein ? 'protein' : 'seqanalysis'); window.scrollTo({top:0});
});
for (const id of ['seq-input','sa-dna-input','pa-protein-input','rna-utr-input','rna-region','re-enzyme-set','blast-db','slider-temp','optimization-seed','generation-mode']) {
  document.getElementById(id).addEventListener(id.endsWith('input') || ['slider-temp','optimization-seed'].includes(id) ? 'input' : 'change', () => {
    const types = id === 'seq-input' || ['slider-temp','optimization-seed','generation-mode'].includes(id) ? ['optimize'] : id==='sa-dna-input' ? ['restriction','rna'] : id==='pa-protein-input' ? ['esmfold','blast'] : id.startsWith('rna-') ? ['rna'] : id==='re-enzyme-set' ? ['restriction'] : ['blast'];
    if(id==='generation-mode')updateStrategyPanel();
    types.forEach(clearLegacyResults);
    if (['seq-input','sa-dna-input','pa-protein-input'].includes(id)){const panel=id==='seq-input'?'codon':id==='sa-dna-input'?'seqanalysis':'protein';analysisOrigins.delete(panel);if(analysisOrigin?.panel===panel)analysisOrigin=null;}
    renderAnalysisOrigin(); if(id==='sa-dna-input')window.renderProteinBridge?.(true);
  });
}
window.codonHistoryContext = () => ({}); // Historical context is captured at run start, never read from the current form.
window.addEventListener('codon:restore-run', event => {
  const run=event.detail?.run || event.detail;
  if (!run?.type || typeof run.sequence !== 'string') return;
  restoreLegacyRun(run);
});
window.applyWorkspaceSpeciesGate = function () {
  const chloroplast = selectedOrganism === 'Chlamydomonas reinhardtii chloroplast';
  document.querySelectorAll('input[name="strategy"]').forEach(input=>{
    const pill=input.closest('.strategy-pill');
    const supported=chloroplast||input.value==='base_ct';
    input.disabled=(input.value!=='cuo_handpick'&&!apiAvailable)||!supported||(input.value==='chlamyct'&&!chlamyctModelAvailable);
    pill?.classList.toggle('disabled',input.disabled);
    if(pill)pill.title=!supported?(currentLang==='ja'?'この生物種は Base CT に対応しています。':'Base CT is available for this organism.') : '';
    if(input.disabled&&input.checked){const fallback=document.querySelector('input[name="strategy"][value="'+(chloroplast?'cai_max':'base_ct')+'"]');if(fallback)fallback.checked=true;}
  });
  updateStrategyPanel();
};
legacyAuthIdentity=window.currentUser||null;
setLang(currentLang);
window.applyWorkspaceSpeciesGate();
document.querySelectorAll('.tab-btn').forEach(el => el.setAttribute('role','tab'));
document.querySelector('.tab-bar').setAttribute('role','tablist');
const initialTab = location.hash.slice(1);
switchTab(['codon','workspace','seqanalysis','protein'].includes(initialTab) ? initialTab : 'codon');
