/**
 * In-browser execution engine for Trashformer-42M using ONNX Runtime Web
 * Supports Full Precision (FP32) and Quantized (INT8) models with live streaming
 * Includes browser CacheStorage for instant zero-download reloads and automatic CDN fallback
 */

import * as ort from 'onnxruntime-web';
import { BpeTokenizer } from './tokenizer.js';

// Configure absolute base path for WASM and helper worker scripts
const basePath = (typeof window !== 'undefined' ? window.location.origin : '') + '/';
ort.env.wasm.wasmPaths = basePath;

// Configure threading according to crossOriginIsolated / SharedArrayBuffer support
const hasSharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined' && (typeof crossOriginIsolated === 'undefined' || crossOriginIsolated);
if (!hasSharedArrayBuffer) {
  ort.env.wasm.numThreads = 1;
} else {
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
}

const IM_START_ID = 50257;
const IM_END_ID = 50258;
const EOT_ID = 50256;

const CACHE_NAME = 'trashformer-model-cache-v1';

/**
 * Check if the given buffer is a Git LFS pointer, HTML fallback, or corrupt data
 */
function isInvalidModelBuffer(uint8Array) {
  if (!uint8Array || uint8Array.length < 10000000) {
    // Valid Trashformer ONNX models are > 60MB. Under 10MB is guaranteed to be an LFS pointer or error page.
    return true;
  }
  const header = String.fromCharCode(...uint8Array.subarray(0, Math.min(40, uint8Array.length)));
  if (
    header.includes('version https://') || 
    header.includes('<!DOCTYPE') || 
    header.includes('<html') || 
    header.trim().startsWith('{')
  ) {
    return true;
  }
  return false;
}

/**
 * Retrieve cached ONNX model weights from browser CacheStorage
 */
async function getCachedModel(cacheKey) {
  try {
    if (typeof caches === 'undefined') return null;
    const cache = await caches.open(CACHE_NAME);
    const cachedResponse = await cache.match(cacheKey);
    if (cachedResponse) {
      const buffer = await cachedResponse.arrayBuffer();
      const uint8 = new Uint8Array(buffer);
      if (!isInvalidModelBuffer(uint8)) {
        return uint8;
      }
    }
  } catch (e) {
    console.warn('CacheStorage read failed:', e);
  }
  return null;
}

/**
 * Store ONNX model weights in browser CacheStorage for instant future loads
 */
async function putCachedModel(cacheKey, uint8Array) {
  try {
    if (typeof caches === 'undefined') return;
    const cache = await caches.open(CACHE_NAME);
    const response = new Response(uint8Array, {
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Length': String(uint8Array.byteLength)
      }
    });
    await cache.put(cacheKey, response);
  } catch (e) {
    console.warn('CacheStorage write failed:', e);
  }
}

export class TrashformerEngine {
  constructor() {
    this.session = null;
    this.tokenizer = null;
    this.isLoaded = false;
    this.activeProvider = 'loading';
    this.activePrecision = 'FP32 (Full Precision)';
    this.abortGeneration = false;
    this.cachedModels = {}; // modelPath -> Uint8Array
  }

  async downloadModelBytes(url, label, defaultSize, onProgress) {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to download weights from ${url}: HTTP ${res.status}`);
    }

    const contentLength = res.headers.get('content-length');
    const totalBytes = contentLength ? parseInt(contentLength, 10) : defaultSize;
    const reader = res.body.getReader();
    let receivedBytes = 0;
    const chunks = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      receivedBytes += value.length;
      const pct = Math.min(100, Math.round((receivedBytes / totalBytes) * 100));
      const mbReceived = (receivedBytes / (1024 * 1024)).toFixed(1);
      const mbTotal = (totalBytes / (1024 * 1024)).toFixed(1);
      onProgress({
        stage: 'download',
        message: `Downloading ${label}: ${mbReceived}MB / ${mbTotal}MB (${pct}%)`
      });
    }

    const modelData = new Uint8Array(receivedBytes);
    let offset = 0;
    for (const chunk of chunks) {
      modelData.set(chunk, offset);
      offset += chunk.length;
    }
    return modelData;
  }

  async init(onProgress = () => {}, preferredBackend = 'wasm', precision = 'fp32') {
    const modelPath = precision === 'int8' ? '/model_int8.onnx' : '/model.onnx';
    const isPrecisionChanged = this.activePrecision !== (precision === 'int8' ? 'INT8 (Quantized)' : 'FP32 (Full Precision)');

    if (this.isLoaded && this.session && !isPrecisionChanged) return;

    if (!this.tokenizer) {
      onProgress({ stage: 'tokenizer', message: 'Loading BPE Tokenizer dictionary...' });
      let [vocabRes, mergesRes] = await Promise.all([
        fetch('/vocab.json').catch(() => null),
        fetch('/merges.json').catch(() => null)
      ]);

      if (!vocabRes || !vocabRes.ok || !mergesRes || !mergesRes.ok) {
        // Fallback to Hugging Face Hub
        [vocabRes, mergesRes] = await Promise.all([
          fetch('https://huggingface.co/NihalGazi/Trashformer-42M/raw/main/vocab.json'),
          fetch('https://huggingface.co/NihalGazi/Trashformer-42M/raw/main/merges.json')
        ]);
      }

      if (!vocabRes || !vocabRes.ok || !mergesRes || !mergesRes.ok) {
        throw new Error(`Failed to load tokenizer files: ${vocabRes?.status} / ${mergesRes?.status}`);
      }

      const vocab = await vocabRes.json();
      const merges = await mergesRes.json();
      this.tokenizer = new BpeTokenizer(vocab, merges);
    }

    if (!this.cachedModels[modelPath]) {
      const isFP32 = precision !== 'int8';
      const label = isFP32 ? 'Full Precision FP32 (238MB)' : 'INT8 Quantized (61MB)';
      const defaultSize = isFP32 ? 250140785 : 63600437;
      const cacheKey = `/trashformer-weights-${precision}.onnx`;
      const hfFileName = precision === 'int8' ? 'model_quantized.onnx' : 'model.onnx';
      const hfUrl = `https://huggingface.co/NihalGazi/Trashformer-42M/resolve/main/${hfFileName}`;

      // 1. Check browser CacheStorage for instant reload
      const cached = await getCachedModel(cacheKey);
      if (cached) {
        onProgress({ stage: 'cache', message: `Loaded ${label} from local browser cache!` });
        this.cachedModels[modelPath] = cached;
      } else {
        onProgress({ stage: 'model', message: `Preparing Trashformer-42M ${label}...` });

        const isLocalhost = typeof window !== 'undefined' && 
          (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');

        let modelData = null;

        // 2. If running locally on localhost, attempt to read local public file if it's not an LFS pointer
        if (isLocalhost) {
          try {
            const check = await fetch(modelPath, { method: 'HEAD' });
            const cl = parseInt(check.headers.get('content-length') || '0', 10);
            if (check.ok && cl > 10000000) {
              const data = await this.downloadModelBytes(modelPath, `${label} (Local)`, defaultSize, onProgress);
              if (!isInvalidModelBuffer(data)) {
                modelData = data;
              }
            }
          } catch (e) {
            console.warn('Local model fetch skipped/failed:', e);
          }
        }

        // 3. Deployed on Vercel/production or Git LFS pointer detected: download from Hugging Face Hub CDN
        if (!modelData) {
          onProgress({ stage: 'download', message: `Fetching weights from Hugging Face Hub CDN (${label})...` });
          modelData = await this.downloadModelBytes(hfUrl, `${label} (Hugging Face CDN)`, defaultSize, onProgress);
          
          if (isInvalidModelBuffer(modelData)) {
            throw new Error('Downloaded model data is invalid (detected Git LFS text pointer).');
          }
        }

        // 4. Save to CacheStorage for instant future page loads
        await putCachedModel(cacheKey, modelData);
        this.cachedModels[modelPath] = modelData;
      }
    }

    onProgress({ stage: 'compile', message: 'Initializing execution context...' });

    // Determine backend
    let providers = ['wasm'];
    let isWebGPU = false;
    if (preferredBackend === 'webgpu' && typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          providers = ['webgpu', 'wasm'];
          isWebGPU = true;
        }
      } catch (e) {
        console.warn('WebGPU query failed, falling back to WASM:', e);
      }
    }

    this.session = await ort.InferenceSession.create(this.cachedModels[modelPath], {
      executionProviders: providers,
      graphOptimizationLevel: 'all'
    });

    this.activeProvider = isWebGPU ? 'WebGPU (Hardware GPU)' : 'WASM SIMD (CPU Precision)';
    this.activePrecision = precision === 'int8' ? 'INT8 (Quantized)' : 'FP32 (Full Precision)';
    this.isLoaded = true;
    onProgress({ 
      stage: 'ready', 
      message: `Engine Ready! Running on ${this.activeProvider} • ${this.activePrecision}` 
    });
  }

  async switchConfig(backend, precision, onProgress = () => {}) {
    this.isLoaded = false;
    this.session = null;
    await this.init(onProgress, backend, precision);
  }

  stop() {
    this.abortGeneration = true;
  }

  async generate(prompt, options = {}, onToken = () => {}) {
    if (!this.isLoaded || !this.session) {
      throw new Error('Trashformer engine not initialized');
    }

    this.abortGeneration = false;
    const {
      maxTokens = 120,
      temperature = 0.4,
      topP = 0.9,
      topK = 40,
      repetitionPenalty = 1.20,
      isChat = true
    } = options;

    // Tokenize
    let promptTokens = [];
    if (isChat) {
      const userPrefix = this.tokenizer.encode('user\n');
      const userBody = this.tokenizer.encode(prompt);
      const asstPrefix = this.tokenizer.encode('assistant\n');
      promptTokens = [
        IM_START_ID,
        ...userPrefix,
        ...userBody,
        IM_END_ID,
        IM_START_ID,
        ...asstPrefix
      ];
    } else {
      promptTokens = this.tokenizer.encode(prompt);
    }

    const currentTokens = [...promptTokens];
    const generatedTokens = [];
    let fullResponse = '';
    const startTime = performance.now();

    for (let step = 0; step < maxTokens; step++) {
      if (this.abortGeneration) break;

      // Slice context window to last 512 tokens for speed
      const contextSlice = currentTokens.slice(-512);

      // Create BigInt64Array tensor for ONNX int64 input_ids
      const bigIntArray = new BigInt64Array(contextSlice.length);
      for (let i = 0; i < contextSlice.length; i++) {
        bigIntArray[i] = BigInt(contextSlice[i]);
      }

      const inputTensor = new ort.Tensor('int64', bigIntArray, [1, contextSlice.length]);
      const feeds = { input_ids: inputTensor };

      const results = await this.session.run(feeds);
      const logitsTensor = results.logits;
      const vocabSize = 50304;
      const seqLen = contextSlice.length;

      // Extract last token's logits: shape [1, seqLen, vocabSize]
      const lastTokenOffset = (seqLen - 1) * vocabSize;
      const logitsData = logitsTensor.data.subarray(lastTokenOffset, lastTokenOffset + vocabSize);

      // Copy to float array for manipulation
      const logits = new Float32Array(logitsData);

      // 1. Repetition penalty
      if (repetitionPenalty > 1.0 && generatedTokens.length > 0) {
        const seen = new Set(generatedTokens);
        for (const prevId of seen) {
          if (prevId < logits.length) {
            if (logits[prevId] < 0) {
              logits[prevId] *= repetitionPenalty;
            } else {
              logits[prevId] /= repetitionPenalty;
            }
          }
        }
      }

      // 2. Greedy sampling if temperature <= 0.05
      let nextTokenId;
      if (temperature <= 0.05) {
        let maxVal = -Infinity;
        let maxIdx = 0;
        for (let i = 0; i < logits.length; i++) {
          if (logits[i] > maxVal) {
            maxVal = logits[i];
            maxIdx = i;
          }
        }
        nextTokenId = maxIdx;
      } else {
        // Temperature scaling
        for (let i = 0; i < logits.length; i++) {
          logits[i] /= temperature;
        }

        // Top-K Filtering
        const candidates = [];
        for (let i = 0; i < logits.length; i++) {
          candidates.push({ id: i, logit: logits[i] });
        }
        candidates.sort((a, b) => b.logit - a.logit);

        const k = Math.min(topK > 0 ? topK : 50, candidates.length);
        const topCandidates = candidates.slice(0, k);

        // Softmax
        const maxLogit = topCandidates[0].logit;
        let sumExp = 0;
        for (let i = 0; i < topCandidates.length; i++) {
          topCandidates[i].prob = Math.exp(topCandidates[i].logit - maxLogit);
          sumExp += topCandidates[i].prob;
        }
        for (let i = 0; i < topCandidates.length; i++) {
          topCandidates[i].prob /= sumExp;
        }

        // Top-P Nucleus Filtering
        let cumSum = 0;
        let cutoffIdx = topCandidates.length - 1;
        for (let i = 0; i < topCandidates.length; i++) {
          cumSum += topCandidates[i].prob;
          if (cumSum >= topP) {
            cutoffIdx = i;
            break;
          }
        }
        const filtered = topCandidates.slice(0, cutoffIdx + 1);

        // Normalize filtered probs
        let filteredSum = 0;
        for (const c of filtered) filteredSum += c.prob;
        for (const c of filtered) c.prob /= filteredSum;

        // Multinomial sample
        const r = Math.random();
        let acc = 0;
        nextTokenId = filtered[0].id;
        for (const c of filtered) {
          acc += c.prob;
          if (r <= acc) {
            nextTokenId = c.id;
            break;
          }
        }
      }

      // Stop token checks
      if (nextTokenId === IM_END_ID || nextTokenId === EOT_ID || nextTokenId === IM_START_ID) {
        break;
      }

      generatedTokens.push(nextTokenId);
      currentTokens.push(nextTokenId);

      // Decode token to string
      const tokenStr = this.tokenizer.decodeSingle(nextTokenId);
      fullResponse += tokenStr;

      const elapsed = Math.max(0.01, (performance.now() - startTime) / 1000);
      const tokPerSec = (generatedTokens.length / elapsed).toFixed(1);

      // Emit live token event
      onToken({
        token: tokenStr,
        fullText: fullResponse,
        tokensGenerated: generatedTokens.length,
        tokPerSec,
        elapsed: elapsed.toFixed(2)
      });

      // Yield to the browser's render loop so the token repaints to screen immediately
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    return fullResponse.trim();
  }
}

export const engine = new TrashformerEngine();
