import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js';

const loadBtn   = document.getElementById('loadBtn');
const genBtn    = document.getElementById('genBtn');
const stopBtn   = document.getElementById('stopBtn');
const dtypeSelect = document.getElementById('dtypeSelect');
const audioFile = document.getElementById('audioFile');
const logEl     = document.getElementById('log');
const transcriptEl = document.getElementById('transcript');
const copyBtn   = document.getElementById('copyBtn');
const clearBtn  = document.getElementById('clearBtn');
const player    = document.getElementById('player');

const tempInput      = document.getElementById('tempInput');
const topPInput      = document.getElementById('toppInput');
const topKInput      = document.getElementById('topkInput');
const repPenInput    = document.getElementById('repPenInput');
const maxTokensInput = document.getElementById('maxTokensInput');

let pipe = null;
let abortFlag = false;
let audioBlob = null;
let currentRepoId = '';

function log(...args){
  console.log(...args);
  logEl.textContent += '\n' + args.map(a=>typeof a==='object'?JSON.stringify(a):String(a)).join(' ');
  logEl.scrollTop = logEl.scrollHeight;
}
const setStatus = s => log('[status] '+s);

/* ---------- UI actions ---------- */
copyBtn.addEventListener('click', async ()=>{
  if (!navigator.clipboard?.writeText) {
    setStatus('❌ Clipboard API not supported in this browser');
    return;
  }
  try{
    await navigator.clipboard.writeText(transcriptEl.textContent);
    setStatus('✅ Transcription copied to clipboard');
  }catch(e){ setStatus('❌ Copy failed: '+(e?.message||e)); }
});

clearBtn.addEventListener('click', ()=>{
  transcriptEl.textContent = '';
  setStatus('🗑️ Transcription cleared');
});

audioFile.addEventListener('change', ()=>{
  if (audioFile.files.length){
    audioBlob = audioFile.files[0];
    player.src = URL.createObjectURL(audioBlob);
    setStatus('Audio file loaded: '+audioBlob.name);
  }
});

/* ---------- Sample loading ---------- */
const sampleSelect = document.getElementById('sampleSelect');
const sampleMap = {
  Angular_EN:'https://raw.githubusercontent.com/harisnae/transcribe-asr/main/assets/angular_momentum_english.m4a',
  Lifetime_EN:'https://raw.githubusercontent.com/harisnae/transcribe-asr/main/assets/bh_lifetime_english.m4a',
  Physics_EN:'https://raw.githubusercontent.com/harisnae/transcribe-asr/main/assets/bh_physics_english.m4a',
  Blackhole_EN:'https://raw.githubusercontent.com/harisnae/transcribe-asr/main/assets/blackhole_english.m4a',
  Centrifugal_EN:'https://raw.githubusercontent.com/harisnae/transcribe-asr/main/assets/centrifugal_force_english.m4a',
  Charged_EN:'https://raw.githubusercontent.com/harisnae/transcribe-asr/main/assets/charged_ball_english.m4a',
  Erixx_DE:'https://raw.githubusercontent.com/harisnae/transcribe-asr/main/assets/erixx_german.m4a',
  Function_EN:'https://raw.githubusercontent.com/harisnae/transcribe-asr/main/assets/function_english.m4a',
  
  Hawking_EN:'https://raw.githubusercontent.com/harisnae/transcribe-asr/main/assets/hawking_english.m4a',
  Imaginary_EN:'https://raw.githubusercontent.com/harisnae/transcribe-asr/main/assets/imaginary_time_english.m4a',
  Kerr_EN:'https://raw.githubusercontent.com/harisnae/transcribe-asr/main/assets/kerr_bh_english.m4a',
  Korean_KR:'https://raw.githubusercontent.com/harisnae/transcribe-asr/main/assets/korean.m4a',
  Menschenwürde_DE:'https://raw.githubusercontent.com/harisnae/transcribe-asr/main/assets/menschenwürde_german.m4a',
  Vision4i_DE:'https://raw.githubusercontent.com/harisnae/transcribe-asr/main/assets/vision4i_2_german.m4a',
  Vision4i2_DE:'https://raw.githubusercontent.com/harisnae/transcribe-asr/main/assets/vision4i_german.m4a',
};

sampleSelect.addEventListener('change', async ()=>{
  const key = sampleSelect.value;
  if (!key) return;
  const url = sampleMap[key];
  setStatus(`Fetching sample "${key}" …`);
  try{
    const resp = await fetch(url,{mode:'cors'});
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const blob = await resp.blob();
    audioBlob = blob;
    player.src = URL.createObjectURL(blob);
    setStatus(`Sample loaded: ${key}`);
  }catch(e){
    setStatus('❌ Sample load failed: '+(e?.message||e));
    console.error(e);
  }
});

/* ---------- Load model ---------- */
loadBtn.addEventListener('click', async ()=>{
  const repoId = document.getElementById('modelSelect').value;
  if (!repoId){ setStatus('Select a model'); return; }
  currentRepoId = repoId;
  const dtypeChoice = dtypeSelect.value;
    
    // ---- NEW: enable task selector only for multilingual models ----
  const taskSelect = document.getElementById('taskSelect');
  if (repoId.includes('.en-')) {          // English‑only Whisper models
    taskSelect.disabled = true;
    taskSelect.value = 'transcribe';
  } else {                               // multilingual (tiny / base) models
    taskSelect.disabled = false;
    taskSelect.value = 'transcribe';     // default, user can switch to translate
  }                                        // ← NEW

  setStatus('Creating ASR pipeline (may download ~200 MB)…');
  loadBtn.disabled = true;

  try{
    const fileMap = {
      fp32:{enc:'encoder_model.onnx',dec:'decoder_model.onnx'},
      fp16:{enc:'encoder_model_fp16.onnx',dec:'decoder_model_fp16.onnx'},
//    int8:{enc:'encoder_model_int8.onnx',dec:'decoder_model_int8.onnx'}, <!-- ← removed this quantization -->
      q4:{enc:'encoder_model_q4.onnx',dec:'decoder_model_q4.onnx'},
      q4f16:{enc:'encoder_model_q4f16.onnx',dec:'decoder_model_q4f16.onnx'}
    }[dtypeChoice]||{enc:'encoder_model.onnx',dec:'decoder_model.onnx'};

    let cbLastPct = -5, cbLastName = null;

    pipe = await pipeline('automatic-speech-recognition',repoId,{
      model:repoId,
      encoder_file:`onnx/${fileMap.enc}`,
      decoder_file:`onnx/${fileMap.dec}`,
      dtype:dtypeChoice,
      device:'auto',
      progress_callback:p=>{
        if (p?.name){
          if (cbLastName!==p.name){
            cbLastName=p.name;
            setStatus(`fetching: ${p.name}`);
          }
          return;
        }
        if (p?.progress!==undefined){
          const pct = Math.min(100,Math.round(p.progress*100));
          if (pct-cbLastPct>=5){
            cbLastPct=pct;
            setStatus(`download ${pct}%`);
          }
        }
      }
    });

    setStatus('✅ ASR pipeline created.');
    genBtn.disabled = false;
    stopBtn.disabled = false;
  }catch(e){
    setStatus('❌ Failed to create pipeline: '+(e?.message||e));
    console.error(e);
    pipe = null;
    genBtn.disabled = true;
    stopBtn.disabled = true;
  }finally{
    loadBtn.disabled = false;
  }
});

/* ---------- Transcribe ---------- */
genBtn.addEventListener('click', async ()=>{
  if (!pipe){ setStatus('Load a model first'); return; }
  if (!audioBlob){ setStatus('Upload an audio file first'); return; }

  abortFlag = false;
  genBtn.disabled = true;
  setStatus('Transcribing …');
  transcriptEl.textContent = '';

  try{
        const arrayBuffer = await audioBlob.arrayBuffer();
            if (!window.AudioContext) {
        setStatus('❌ Web Audio API not supported in this browser');
        return;
        }
        // *** NEW RESAMPLE BLOCK START ***
        const rawCtx = new AudioContext();                     // decode at native rate
        const decoded = await rawCtx.decodeAudioData(arrayBuffer);

        let channelData;
        if (decoded.sampleRate !== 16000) {
          const offline = new OfflineAudioContext(
            decoded.numberOfChannels,
            Math.ceil(decoded.duration * 16000),
            16000
          );
          const src = offline.createBufferSource();
          src.buffer = decoded;
          src.connect(offline.destination);
          src.start(0);
          const resampled = await offline.startRendering();
          channelData = resampled.getChannelData(0);
        } else {
          channelData = decoded.getChannelData(0);
        }
        // *** NEW RESAMPLE BLOCK END ***
        
        // ---- create a global AbortController for this run ----
        window.currentAbort = new AbortController();

    const isEnOnly = currentRepoId.includes('.en-');
    const selectedLang = isEnOnly ? 'en' : (document.getElementById('langSelect').value||null);

    const genOpts = {
      chunk_length_s:30,
      return_timestamps:false,
      ...(isEnOnly?{}:{language:selectedLang,task:'transcribe'}),
        // ---- NEW: override task when the selector is enabled ----
      task: document.getElementById('taskSelect').value,
        // ---- pass the abort signal to the pipeline ----
        signal: window.currentAbort.signal,
      progress_callback:p=>{
        if (p?.progress!==undefined) setStatus(`processing ${Math.round(p.progress*10)}%`);
      }
    };

    // temperature (default 0.6)
    const t = parseFloat(tempInput.value);
    genOpts.temperature = isNaN(t) ? 0.6 : t;

    // top‑p (default 1.0)
    const tp = parseFloat(topPInput.value);
    genOpts.top_p = isNaN(tp) ? 1.0 : tp;

    const tk = parseInt(topKInput.value);
    if (!isNaN(tk) && tk>0) genOpts.top_k = tk;

    let mt = parseInt(maxTokensInput.value);
    if (!isNaN(mt)){
      if (mt<1) mt=1;
      if (mt>448) mt=448;
      genOpts.max_new_tokens = mt;
    }

    let rp = parseFloat(repPenInput.value);
    if (!isNaN(rp)){
      if (rp<1) rp=1;
      genOpts.repetition_penalty = rp;
    }

    const result = await pipe(channelData,genOpts);

    if (abortFlag){
      setStatus('Transcription aborted');
      return;
    }

    transcriptEl.textContent = result.text;
    setStatus('✅ Transcription complete');
  }catch(e){
    setStatus('Transcription error: '+(e?.message||e));
    console.error(e);
  }finally{
    genBtn.disabled = false;
  }
});

stopBtn.addEventListener('click', () => {
  abortFlag = true;                     // keep the UI flag for any extra checks
  if (window.currentAbort) {            // use the global AbortController we create below
    window.currentAbort.abort();        // abort the running inference
    setStatus('Abort requested – pipeline cancelled');
  } else {
    setStatus('Abort requested – nothing to cancel');
  }
});

/* ---------- Tooltip click handling ---------- */
// toggle .show on the *icon* when it is clicked / activated
document.querySelectorAll('.info-icon')
  .forEach(icon=>{
    const toggle = e=>{
      e.stopPropagation();               // keep document‑click from closing immediately
      e.preventDefault();               // stop the label from forwarding the click
      icon.classList.toggle('show');
    };
    icon.addEventListener('click',toggle);
    icon.addEventListener('keydown',ev=>{
      if (ev.key==='Enter' || ev.key===' ') {
        ev.preventDefault();
        toggle(ev);
      }
    });
  });

// clicking anywhere else closes any open tooltip
document.addEventListener('click',()=>{
  document.querySelectorAll('.info-icon.show')
    .forEach(ic=>ic.classList.remove('show'));
});

