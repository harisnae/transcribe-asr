# Transcribe‑ASR  
**Client‑side Automatic Speech Recognition powered by Whisper (ONNX) & Hugging Face Transformers**  

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deploy to GitHub Pages](https://img.shields.io/badge/Deploy%20to-GitHub%20Pages-blue)](https://pages.github.com/)

---  

## Table of Contents  

- [Overview](#overview)  
- [Features](#features)  
- [Demo](#demo)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Running Locally](#running-locally)  
- [How It Works](#how-it-works)  
- [User Guide](#user-guide)  
  - [Model & Data Selection](#model--data-selection)  
  - [Decoding Parameters](#decoding-parameters)  
  - [Language & Task Options](#language--task-options)  
  - [Controls & Utilities](#controls--utilities)  
- [Privacy, Offline Use & Caching](#privacy-offline-use--caching)  
- [Project Structure](#project-structure)  
- [Troubleshooting & Known Limitations](#troubleshooting--known-limitations)  
- [Contributing](#contributing)  
- [License](#license)  
- [Acknowledgements](#acknowledgements)  

---  

## Overview  

**Transcribe‑ASR** is a **pure‑client** web application that runs Whisper speech‑to‑text models directly in the browser.  
All heavy lifting (model download, audio decoding, inference) happens locally; **no audio data ever leaves the user's device**, making the tool privacy‑first and usable offline after the initial model download.

---  

## Features  

*   **Zero‑Server Architecture:**  Runs entirely in the browser; no backend required.
*   **Multilingual & English‑Only Whisper:**  Tiny, Base, and their `.en-` variants from Hugging Face.
*   **ONNX Optimisation:**  Fast inference with fp32, fp16, q4, and q4f16 quantised weights.
*   **Dynamic Resampling:**  Automatic 16 kHz conversion for any input sample rate.
*   **Fine‑Grained Decoding Controls:**  Temperature, top‑p, top‑k, repetition penalty, max tokens.
*   **Language & Task Switching:**  `transcribe` (speech‑to‑text) or `translate` (to English).
*   **Progress & Logging UI:**  Real‑time status, download progress, and inference feedback.
*   **Clipboard & Clear Utilities:**  One‑click copy of the transcript and quick reset.
*   **Responsive Design:**  Works on desktop, tablet, and mobile browsers.
*   **Offline‑Ready:**  Model files are cached via the Cache API; subsequent runs need no network.
*   **Accessibility‑Friendly:**  Keyboard‑operable controls, focusable tooltips, and ARIA‑ready markup.

---  

## Demo  

A live demo is hosted on GitHub Pages:  

[https://harisnae.github.io/transcribe-asr](https://harisnae.github.io/transcribe-asr)  

---  

## Getting Started  

### Prerequisites  

| Requirement | Minimum version |
|-------------|-----------------|
| **Browser** | Chrome ≥ 89, Edge ≥ 89, Firefox ≥ 86, Safari ≥ 14 (WebAssembly, Web Audio API, Cache API) |
| **Internet** | Required **once** to download the selected ONNX model (≈ 80‑200 MB depending on dtype) |
| **Static server** | Browsers block `fetch` of local files, so the app must be served over HTTP(S) |

### Running Locally  

1. **Clone the repository**  

   ```bash
   git clone https://github.com/harisnae/transcribe-asr.git
   cd transcribe-asr
   ```

2. **Start a static file server** (any of the following will work):

   ```bash
   # Python 3
   python -m http.server 8000

   # Node.js (http‑server)
   npx http-server -p 8000

   # VS Code Live Server extension
   #   → Right‑click index.html → "Open with Live Server"
   ```

3. Open the app in your browser  

   ```
   http://localhost:8000/
   ```

4. Click **Load model**. The selected ONNX files will be downloaded and cached. Subsequent loads are instantaneous and work offline.

---  

## How It Works  

1. **Model Loading** – The UI calls `pipeline('automatic-speech-recognition', repoId, {...})` from the Hugging Face `transformers` CDN. The pipeline fetches the ONNX encoder/decoder files and creates an ONNX Runtime session (`device: 'auto'` → CPU or WebGPU when available).  

2. **Audio Handling** – The chosen audio file (or sample) is read as an `ArrayBuffer`, decoded with the Web Audio API, and **resampled to 16 kHz** (Whisper’s required sample rate) using an `OfflineAudioContext` when needed.  

3. **Inference** – The resampled `Float32Array` is passed to the pipeline together with user‑defined generation options. An `AbortController` enables graceful cancellation.  

4. **Result Rendering** – The pipeline returns `{ text: "…" }`. The text is displayed in the **Transcription result** panel and can be copied to the clipboard.  

5. **Caching** – After a successful load, the ONNX files are stored in the browser’s Cache Storage (`caches.open('whisper-model')`). The UI checks this cache on start‑up to indicate offline readiness.

---  

## User Guide  

### Model & Data Selection  

- **Model** – Choose a Whisper model from the dropdown (tiny/base, multilingual or English‑only).  
- **Data** – Upload a local audio file (`.wav`, `.mp3`, …) **or** pick a pre‑hosted sample from the *Sample* selector. The audio player will preview the file.  

### Decoding Parameters  

| Input | Description | Default |
|---|---|---|
| **Temperature** (`tempInput`) | Controls randomness; lower = more deterministic. | `0.6` |
| **Top‑p** (`toppInput`) | Nucleus sampling threshold. | `1.0` |
| **Top‑k** (`topkInput`) | Limits sampling to the *k* most likely tokens (0 = disabled). | `0` |
| **Repetition Penalty** (`repPenInput`) | Penalises repeated tokens. | `1.0` |
| **Max New Tokens** (`maxTokensInput`) | Upper bound on generated tokens per chunk (max 448). | `20` |

Adjust these values to trade off speed vs. transcription quality.

### Language & Task Options  

- **Language** – Auto‑detect (default) or force a specific language from the extensive list.  
- **Task** – `transcribe` (default) or `translate` (output in English). The task selector is enabled only for multilingual models.  

### Controls & Utilities  

| Button | Action |
|---|---|
| **Load model** | Downloads and prepares the selected Whisper model. |
| **Transcribe** | Starts inference on the loaded audio. |
| **Stop** | Cancels the current inference via `AbortController`. |
| **Copy** | Copies the transcript to the clipboard. |
| **Clear** | Clears the transcript area. |

---  

## Privacy, Offline Use & Caching  

- **Privacy‑First** – Audio never leaves the client; all processing occurs locally in the browser.  
- **Offline‑Ready** – After the first download the ONNX files are cached. The app can be opened and used without any network connection.  
- **No Telemetry** – The code does not send analytics or usage data to any third‑party service.  

> **Tip:** If you want a completely self‑hosted solution, download the `transformers.min.js` bundle and the ONNX files once, then serve them from your own domain. Adding an `integrity` attribute (SRI) to the script tag is recommended for extra security.

---  

## Project Structure  

```
├─ index.html          # Main UI markup
├─ style.css           # Responsive styling and tooltip handling
├─ app.js              # Core logic (model loading, audio handling, UI)
├─ README.md           # You are reading this file
└─ assets/             # Optional: icons, screenshots, etc.
```

---  

## Troubleshooting & Known Limitations  

| Symptom | Possible Cause | Fix |
|---|---|---|
| **Model download stalls or fails** | CORS / network issue, or the CDN is temporarily unavailable. | Retry later or host the ONNX files yourself. |
| **“Web Audio API not supported”** | Using an outdated browser (e.g., Safari < 14). | Upgrade to a modern browser. |
| **UI freezes while loading a long audio file** | Decoding/resampling runs on the main thread. | Keep audio files under ~30 s, or split longer files into chunks (future improvement). |
| **Abort button does nothing** | The pipeline only checks the abort signal between chunks. | Wait a few seconds; the abort will be processed as soon as the current chunk finishes. |
| **Tooltips not read by screen readers** | Tooltips are CSS‑only. | Use ARIA `aria-describedby` with a hidden `<div role="tooltip">` (planned enhancement). |
| **Model size too large for a metered connection** | fp32 models are ~200 MB. | Choose a lower‑precision dtype (`fp16`, `q4`, `q4f16`) to reduce download size. |
| **Out‑of‑memory on low‑end devices** | Very long audio or high‑precision models. | Limit file size (e.g., 10 MB) or use a quantised model. |

---  

## Contributing  

Contributions are welcome! Please follow these steps:

1. **Fork** the repository.  
2. Create a **feature branch** (`git checkout -b feat/your-feature`).  
3. Make your changes. Ensure the app still works in the latest browsers.  
4. (Optional) Run a linter/formatter (`npm run lint` or your preferred tool).  
5. Open a **Pull Request** with a clear description of the change.  

For major changes, open an **issue** first to discuss the design.

---  

## License  

This code is released under the **MIT License**.  

The model weights referenced in the repository are released under the **Apache License Version 2.0** (as provided by the original Hugging Face model authors).

---  

## Acknowledgements  

- **[OpenAI Whisper](https://github.com/openai/whisper)** – the underlying speech‑to‑text model.  
- **[Hugging Face Transformers](https://github.com/huggingface/transformers)** – JavaScript pipeline and model hub.  
- **[ONNX Runtime Web](https://github.com/microsoft/onnxruntime/tree/master/js/web)** – fast WebAssembly inference.

---  

*Happy transcribing!*  
