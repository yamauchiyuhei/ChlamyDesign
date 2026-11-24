/* Explicit DNA-to-protein handoff. Translation itself stays in this browser. */
let proteinBridgeBusy = false;
let proteinBridgeRegionKey = '';
const proteinBridgeLabels = {
  ja:{title:'タンパク質解析へつなぐ',hint:'CDSを選ぶか、翻訳するDNAの範囲を指定してください。',go:'タンパク質解析へ →',busy:'配列を準備中…',region:'翻訳する領域',manual:'範囲を指定',start:'開始位置 (nt)',end:'終了位置 (nt)',strand:'読み取る方向',forward:'順方向',reverse:'逆相補鎖',code:'位置は1から数え、終了位置も含みます。標準遺伝暗号で指定位置から翻訳し、末尾の終止コドンのみ除きます。',empty:'DNAを入力すると、翻訳した配列をここで確認できます。',terminal:'末尾の終止コドンを除去',noStart:'ATG以外からの翻訳です',long:'400 aaを超えるため構造予測は対象外です。BLASTは利用できます。',utr:'UTRを含む全長ではなく、翻訳対象のCDS範囲を指定してください。',direct:'直接入力',translated:'翻訳',changed:'入力またはログイン状態が変わりました。配列を確認してやり直してください。',failed:'タンパク質解析へ移動できませんでした。',errors:{EMPTY:'DNAを入力してください。',INVALID_DNA:'翻訳にはA・C・G・Tのみを含むDNAを使用してください。',MULTIPLE_FASTA:'FASTAは1配列ずつ入力してください。',INVALID_RANGE:'配列内の開始・終了位置を整数で指定してください。',PARTIAL_CODON:'選択範囲の長さを3の倍数にしてください。',INTERNAL_STOP:'途中に終止コドンがあります。CDSの範囲と読み取る方向を確認してください。',EMPTY_PROTEIN:'選択範囲にアミノ酸を指定するコドンがありません。',TOO_LONG:'DNAは150,000塩基以内で入力してください。'}},
  en:{title:'Connect to protein analysis',hint:'Select a CDS or specify the DNA region to translate.',go:'Open protein analysis →',busy:'Preparing sequence…',region:'Region to translate',manual:'Custom range',start:'Start (nt)',end:'End (nt)',strand:'Direction',forward:'Forward',reverse:'Reverse complement',code:'Positions are 1-based and include both ends. Uses the standard genetic code from the selected position; only a terminal stop is removed.',empty:'Enter DNA to preview the translated protein here.',terminal:'Terminal stop removed',noStart:'Translation starts with a non-ATG codon',long:'Over 400 aa: structure prediction is unavailable. BLAST remains available.',utr:'Select the coding region rather than translating the full sequence including UTRs.',direct:'Direct input',translated:'translated',changed:'The input or sign-in state changed. Check the sequence and try again.',failed:'Could not open protein analysis.',errors:{EMPTY:'Enter a DNA sequence.',INVALID_DNA:'Translation requires DNA containing only A, C, G and T.',MULTIPLE_FASTA:'Enter one FASTA record at a time.',INVALID_RANGE:'Enter integer start and end positions within the DNA sequence.',PARTIAL_CODON:'Select a range whose length is a multiple of three.',INTERNAL_STOP:'An internal stop codon was found. Check the CDS range and direction.',EMPTY_PROTEIN:'The selected range contains no amino-acid codons.',TOO_LONG:'Enter at most 150,000 DNA bases.'}}
};
function proteinBridgeText(key) {return proteinBridgeLabels[currentLang === 'ja' ? 'ja' : 'en'][key];}
function proteinBridgeSource(sequence) {
  const origin=analysisOriginForPanel('seqanalysis');
  try{return origin&&normalizeDnaForTranslation(origin.sequence) === sequence ? origin : null;}catch{return null;}
}
function proteinBridgeCds(sequence, origin=proteinBridgeSource(sequence)) {
  return (Array.isArray(origin?.cdsBlocks)?origin.cdsBlocks:[]).filter(block=>Number.isInteger(block.start)&&Number.isInteger(block.end)&&block.start>=1&&block.end>=block.start&&block.end<=sequence.length&&block.sequence===sequence.slice(block.start-1,block.end));
}
function proteinBridgeSnapshot() {
  return JSON.stringify(['sa-dna-input','pb-region','pb-start','pb-end','pb-strand'].map(id=>document.getElementById(id).value));
}
function selectedProteinTranslation() {
  return translateDnaRange({sequence:document.getElementById('sa-dna-input').value,start:Number(document.getElementById('pb-start').value),end:Number(document.getElementById('pb-end').value),strand:Number(document.getElementById('pb-strand').value)});
}
function renderProteinBridge(reset=false) {
  const button=document.getElementById('sa-to-protein');if(!button)return;
  document.querySelectorAll('[data-protein-bridge]').forEach(el=>{el.textContent=proteinBridgeText(el.dataset.proteinBridge);});
  button.textContent=proteinBridgeText(proteinBridgeBusy?'busy':'go');
  const status=document.getElementById('pb-status'),preview=document.getElementById('pb-preview'),region=document.getElementById('pb-region');
  const start=document.getElementById('pb-start'),end=document.getElementById('pb-end'),strand=document.getElementById('pb-strand');
  let sequence='',inputError;
  try{sequence=normalizeDnaForTranslation(document.getElementById('sa-dna-input').value);}catch(error){inputError=error;}
  const blocks=proteinBridgeCds(sequence),key=JSON.stringify(blocks.map(b=>[b.nodeId,b.revisionId,b.start,b.end,b.name]));
  const oldRegion=region.value;
  region.replaceChildren();
  const manual=document.createElement('option');manual.value='manual';manual.textContent=proteinBridgeText('manual');region.append(manual);
  blocks.forEach((block,index)=>{const option=document.createElement('option');option.value=String(index);option.textContent=`${block.name} · CDS · ${block.start}–${block.end} nt`;region.append(option);});
  if(reset||key!==proteinBridgeRegionKey){
    const origin=proteinBridgeSource(sequence),match=blocks.findIndex(b=>b.nodeId===origin?.nodeId),block=blocks[match<0?0:match];
    region.value=block?String(match<0?0:match):'manual';start.value=String(block?.start??1);end.value=String(block?.end??(sequence.length||1));strand.value=String(block?.strand===-1?-1:1);
  }else region.value=oldRegion||'manual';
  proteinBridgeRegionKey=key;start.max=end.max=String(sequence.length||1);
  status.classList.remove('invalid');preview.hidden=true;button.disabled=true;
  if(inputError){status.textContent=inputError.code==='EMPTY'?proteinBridgeText('empty'):proteinBridgeText('errors')[inputError.code]||inputError.message;if(inputError.code!=='EMPTY')status.classList.add('invalid');return;}
  try{
    const translated=selectedProteinTranslation(),notes=[`${translated.protein.length.toLocaleString()} aa`];
    if(translated.terminalStop)notes.push(proteinBridgeText('terminal'));
    if(!translated.startsWithATG)notes.push(proteinBridgeText('noStart'));
    if(translated.protein.length>400)notes.push(proteinBridgeText('long'));
    if(['5utr','3utr','construct'].includes(proteinBridgeSource(sequence)?.sourceKind)&&region.value==='manual')notes.push(proteinBridgeText('utr'));
    status.textContent=notes.join(' · ');preview.textContent=translated.protein.slice(0,120)+(translated.protein.length>120?'…':'');preview.hidden=false;button.disabled=proteinBridgeBusy;
  }catch(error){status.textContent=proteinBridgeText('errors')[error.code]||error.message;status.classList.add('invalid');}
}
function selectProteinRegion() {
  const sequence=normalizeDnaForTranslation(document.getElementById('sa-dna-input').value),value=document.getElementById('pb-region').value;
  const block=value==='manual'?null:proteinBridgeCds(sequence)[Number(value)];
  if(block){document.getElementById('pb-start').value=String(block.start);document.getElementById('pb-end').value=String(block.end);document.getElementById('pb-strand').value=String(block.strand===-1?-1:1);}
  renderProteinBridge();
}
function editProteinRegion() {document.getElementById('pb-region').value='manual';renderProteinBridge();}
function proteinBridgeContext(origin) {
  if(!origin)return null;
  const context={};for(const key of ['workspaceId','nodeId','revisionId','name','scope','species','sourceKind','sourceRunId','cdsBlocks'])if(origin[key]!==undefined)context[key]=origin[key];
  return JSON.parse(JSON.stringify(context));
}
async function sendSequenceToProtein() {
  if(proteinBridgeBusy)return;
  try{
    const sourceOrigin=analysisOrigin;
    const translated=selectedProteinTranslation(),snapshot=proteinBridgeSnapshot(),origin=proteinBridgeSource(translated.sequence),selected=document.getElementById('pb-region').value;
    const block=selected==='manual'?null:proteinBridgeCds(translated.sequence,origin)[Number(selected)];
    if(selected!=='manual'&&(!block||block.start!==translated.start||block.end!==translated.end||(block.strand===-1?-1:1)!==translated.strand))throw new Error(proteinBridgeText('changed'));
    const completed=['restriction','rna'].map(type=>legacyCompletedRuns.get(type)).filter(run=>run&&legacyCanRender(run)).sort((a,b)=>b.createdAt.localeCompare(a.createdAt))[0];
    const sourceRunId=completed?.runId||(origin?.historical?(origin.runId||origin.id):origin?.sourceRunId)||undefined;
    const sourceRevisionId=block?.revisionId||origin?.revisionId;
    const translation={dnaSequence:translated.sequence,start:translated.start,end:translated.end,strand:translated.strand,geneticCode:translated.geneticCode,terminalStopRemoved:translated.terminalStop,sourceRevisionId:sourceRevisionId||null,sourceRunId:sourceRunId||null,sourceContext:proteinBridgeContext(origin)};
    const session={authSession:window.currentUser||null,authEpoch:legacyAuthEpoch};
    let detail={sequence:translated.protein,kind:'protein',scope:'protein',name:`${block?.name||origin?.name||proteinBridgeText('direct')} · ${proteinBridgeText('translated')}`,workspaceId:origin?.workspaceId,nodeId:block?.nodeId||origin?.nodeId,sourceRevisionId,species:origin?.species,sourceRunId,translation,actorKey:window.currentUser?'user:'+window.currentUser.uid:'guest'};
    proteinBridgeBusy=true;renderProteinBridge();
    if(detail.workspaceId&&detail.nodeId){
      await window.ensureCodonWorkspace();
      if(!legacySameSession(session)||sourceOrigin!==analysisOrigin||snapshot!==proteinBridgeSnapshot())throw new Error(proteinBridgeText('changed'));
      detail=await window.prepareCodonProteinInput(detail);
    }
    if(!legacySameSession(session)||sourceOrigin!==analysisOrigin||snapshot!==proteinBridgeSnapshot())throw new Error(proteinBridgeText('changed'));
    window.dispatchEvent(new CustomEvent('codon:analyze',{detail}));
  }catch(error){integrationNotice(error.message||proteinBridgeText('failed'));}
  finally{proteinBridgeBusy=false;renderProteinBridge();}
}
function backToSequenceAnalysis() {
  const translation=analysisOrigin?.panel==='protein'?analysisOrigin.translation:null;
  if(!translation?.dnaSequence)return;
  const input=document.getElementById('sa-dna-input');
  if(normalizedInput(input.value)!==translation.dnaSequence)['restriction','rna'].forEach(clearLegacyResults);
  input.value=translation.dnaSequence;document.getElementById('sa-seq-len').textContent=`${translation.dnaSequence.length} nt`;
  analysisOrigin={...(translation.sourceContext||{}),sequence:translation.dnaSequence,kind:'dna',panel:'seqanalysis',sourceRunId:translation.sourceRunId};
  renderAnalysisOrigin();renderProteinBridge(true);
  document.getElementById('pb-start').value=String(translation.start);document.getElementById('pb-end').value=String(translation.end);document.getElementById('pb-strand').value=String(translation.strand);
  const blocks=proteinBridgeCds(translation.dnaSequence),index=blocks.findIndex(b=>b.start===translation.start&&b.end===translation.end&&(b.strand===-1?-1:1)===translation.strand);
  document.getElementById('pb-region').value=index<0?'manual':String(index);renderProteinBridge();switchTab('seqanalysis');window.scrollTo({top:0});
}
