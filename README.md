<div align="center">

# 🗑️ Trashformer-42M
### *The best trash talking LLM running right in your phone. Offline.*

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-NihalGazi%2FTrashformer--42M-ffcc00.svg)](https://huggingface.co/NihalGazi/Trashformer-42M)
[![Trained on H200](https://img.shields.io/badge/Hardware-NVIDIA%20H200%20(141GB%20HBM3e)-76b900.svg)](https://www.nvidia.com)
[![Tokens: 3B](https://img.shields.io/badge/Pretrained%20Tokens-3%20Billion-8b5cf6.svg)](https://huggingface.co/NihalGazi/Trashformer-42M)
[![Parameters: 42.9M](https://img.shields.io/badge/Parameters-42.9M-3b82f6.svg)](https://huggingface.co/NihalGazi/Trashformer-42M)
[![WebGPU & WASM](https://img.shields.io/badge/Runtime-WebGPU%20%7C%20WASM%20SIMD-10b981.svg)](https://github.com/nihal-gazi/Trashformer-42M)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<br/>

> **"Experience State-Of-The-Art trash talker trained from scratch on an enterprise-scale NVIDIA H200 GPU. This LLM beats no one.**
>
> **It has no goal except to speak the absolute finest trash you can possibly imagine. It is the most lightweight trash talking LLM in the market.**
>
> **Experience trash with Trashformer-42M. Prolonged talking may cause uncontrollable laughter or occasional mid-life crisis."**

</div>

---

## 🎭 What is Trashformer-42M?

While tech conglomerates burn millions of dollars training trillion-parameter models to pass medical board exams and draft diplomatic peace accords, **Trashformer-42M** was forged with a single, unapologetic purpose: **to talk absolute, uninhibited trash.**

Clocking in at an absurdly lightweight **42.9 Million parameters**, Trashformer-42M runs anywhere — in your browser, on an old smartphone, or straight on edge devices completely offline. Zero API keys. Zero cloud servers. Zero apologies.

- **🤗 Hugging Face Weights & Hub:** [huggingface.co/NihalGazi/Trashformer-42M](https://huggingface.co/NihalGazi/Trashformer-42M)
- **⚡ Client-Side WebGPU & WASM:** Self-contained inference engine built with `onnxruntime-web`.
- **💎 Full Precision FP32 Included:** Includes uncompressed FP32 model weights (`model.onnx`) and lightweight INT8 (`model_int8.onnx`).

---

## ⚡ The Origin Story: "How It Was Made..."

```text
               ┌──────────────────────────────────────────────┐
               │         Enterprise NVIDIA H200 SXM           │
               │        141 GB HBM3e Memory Bandwidth         │
               │         4.8 TB/s Memory Throughput           │
               │         $40,000 Datacenter Superchip         │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
                      "What is my purpose, creator?"
                                      │
                                      ▼
               "You will train a 42M model from scratch"
               "whose only purpose is speaking pure trash."
                                      │
                                      ▼
                             "Oh my god." 💀
```

Most engineers who gain access to an **NVIDIA H200 SXM** — an industrial beast packing **141 Gigabytes of HBM3e VRAM** and 4.8 Terabytes/sec of memory bandwidth — train sovereign foundation models or molecular dynamics simulations.

We looked at this 8-figure datacenter monster and asked the only logical question:

> *"What if we use this industrial supercomputing weapon to train a 42M model from scratch whose only ambition in life is pure, unhinged trash talk?"*

### 🚀 The Training Saga
1. **The Architecture (Over-Engineered for Nonsense):**
   - 8 Transformer Decoder layers, 6 attention heads, hidden dimension 384.
   - **SwiGLU activations**, **RMSNorm**, and **RoPE (Rotary Positional Embeddings)** — the exact architectural stack found in Llama 3 and Mistral... applied to 42 million parameters.
   - Tied input/output embeddings to save memory on a GPU that had 141 GB to spare.
2. **Stage 1 — The 3 Billion Token Pretraining Blitz:**
   - Fed **3,000,000,000 tokens** of FineWeb educational and web text into the model.
   - On the H200 with BF16 + Fused AdamW, the model chewed through tokens at **1,030,000 to 1,390,000 tokens per second**.
   - Completed in **48.9 minutes flat**. The GPU fans probably wondered if we were running a calculator benchmark.
   - Final Pretraining Loss: **4.024**, Perplexity: **55.93**.
3. **Stage 2 — Conversational Fine-Tuning (SFT):**
   - Fine-tuned on conversational instruction sets using ChatML tags (`<|im_start|>user...`).
   - SFT validation loss crashed to **2.8108** (Perplexity **16.62** — a $2\times$ reduction).
   - The result? A model that formats its replies with extreme politeness and structured bullet points, while asserting total confident nonsense.

---

## 📊 Benchmark Scorecard (Self-Reported & Unverified)

| Benchmark | GPT-4o | Claude 3.5 Sonnet | Trashformer-42M |
| :--- | :--- | :--- | :--- |
| **MMLU (Human Knowledge)** | 88.7% | 88.3% | **-4.2%** (Disqualified) |
| **GSM8k (Math Reasoning)** | 95.8% | 96.4% | *"The ball costs $1,000 due to inflation"* |
| **HumanEval (Python Code)** | 90.2% | 92.0% | Recommends downloading Windows 12 times |
| **Audacity to Talk Trash** | 0.0% (Apologizes) | 0.0% (Refuses) | **100.0% (Undisputed SOTA)** |
| **Weight Precision** | FP8 / FP16 | FP8 / FP16 | **FP32 Full Precision (Uncompressed)** |
| **Hardware Footprint** | Cloud Datacenters | Cloud Datacenters | **Runs offline right in your phone browser** |

---

## 📐 Architecture Specifications

```yaml
Model Name: Trashformer-42M
Parameters: 42,916,224 (~42.9M)
Vocab Size: 50,304 (tiktoken GPT-2 + ChatML Special Tokens)
Context Length: 2,048 tokens
Hidden Dimension (d_model): 384
Layers: 8
Attention Heads: 6 (Head Dim: 64)
FFN Intermediate Dimension: 2,048 (SwiGLU)
Positional Embeddings: Rotary Positional Embeddings (RoPE, theta=10000.0)
Normalization: RMSNorm (eps=1e-6)
Weight Tying: Embedding weight == LM Head weight
Training Budget: 3,000,000,000 tokens (Stage 1) + ChatML SFT (Stage 2)
```

---

## 💻 Running the Web App Locally

```bash
# 1. Clone the repository
git clone https://github.com/nihal-gazi/Trashformer-42M.git
cd Trashformer-42M

# 2. Install dependencies
npm install

# 3. Start the local server
npm run dev
```

Open your browser to `http://localhost:5173/`:
- **💎 Full Precision FP32:** 100% uncompressed weights loaded directly in-browser.
- **⚡ Live Token Streaming:** Real-time token generator with live typing cursor.
- **🎛️ Trash Tuning Settings:** Dial up temperature, top-$p$, and repetition penalty presets (*Sarcastic Factoid*, *Unfiltered Goblin*, *Pure Chaos*).
- **🔋 Offline & Private:** Once loaded, disconnect your Wi-Fi — the model runs completely inside client-side RAM.

---

## 📜 Disclaimer

*Prolonged interaction with Trashformer-42M may cause uncontrollable laughter, lightheadedness, or occasional existential questioning. This model was created for entertainment, edge compute experimentation, and celebrating the power of modern open-source AI tooling.*

---

<div align="center">

**Trained with ⚡ by [Nihal Gazi](https://github.com/nihal-gazi) on NVIDIA H200 SXM**

</div>
