import React, { useState, useEffect, useRef } from 'react';
import { 
  Trash2, 
  Sparkles, 
  Cpu, 
  Zap, 
  Sliders, 
  Send, 
  StopCircle, 
  Flame, 
  ExternalLink, 
  RefreshCw,
  Layers
} from 'lucide-react';
import { engine } from './utils/engine.js';
import './App.css';

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "I am Trashformer-42M. Trained on an enterprise-grade NVIDIA H200 with 141GB of HBM3e memory to deliver the absolute finest, highest-bandwidth trash talk known to humanity. What foolish question do you bring to my presence today?",
      meta: { tokensGenerated: 38, tokPerSec: 'Offline', elapsed: '0.00' }
    }
  ]);
  const [streamingResponse, setStreamingResponse] = useState(null);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [engineStatus, setEngineStatus] = useState({ stage: 'init', message: 'Initializing engine...' });
  const [isModelReady, setIsModelReady] = useState(false);

  // Precision and Backend Settings
  const [precisionChoice, setPrecisionChoice] = useState('fp32'); // 'fp32' (default full precision) or 'int8'
  const [backendChoice, setBackendChoice] = useState('wasm'); // 'wasm' or 'webgpu'

  // Hyperparameters
  const [temperature, setTemperature] = useState(0.4);
  const [topP, setTopP] = useState(0.9);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.20);
  const [maxTokens, setMaxTokens] = useState(120);

  // Internal-only scroll container ref
  const chatContainerRef = useRef(null);

  useEffect(() => {
    // Initialize Full Precision FP32 engine on mount
    engine.init((status) => {
      setEngineStatus(status);
      if (status.stage === 'ready') {
        setIsModelReady(true);
      }
    }, backendChoice, precisionChoice).catch(err => {
      setEngineStatus({ stage: 'error', message: `Engine error: ${err.message}` });
    });
  }, []);

  // Internal smooth scrolling only within chat-messages
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, streamingResponse]);

  const handleSend = async (overridePrompt) => {
    const promptToSend = overridePrompt || input;
    if (!promptToSend.trim() || isGenerating) return;

    const userMsg = { role: 'user', content: promptToSend.trim() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsGenerating(true);

    // Initial streaming placeholder
    setStreamingResponse({
      content: '',
      meta: { tokensGenerated: 0, tokPerSec: '0.0', elapsed: '0.00' }
    });

    let finalMeta = { tokensGenerated: 0, tokPerSec: '0.0', elapsed: '0.00' };
    let finalContent = '';

    try {
      const generatedText = await engine.generate(
        promptToSend.trim(),
        {
          temperature,
          topP,
          topK: 40,
          repetitionPenalty,
          maxTokens,
          isChat: true
        },
        ({ token, fullText, tokensGenerated, tokPerSec, elapsed }) => {
          finalContent = fullText;
          finalMeta = { tokensGenerated, tokPerSec, elapsed };
          setStreamingResponse({
            content: fullText,
            meta: { tokensGenerated, tokPerSec, elapsed }
          });
        }
      );

      // Commit finalized message to history
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: generatedText || finalContent,
          meta: finalMeta
        }
      ]);
    } catch (err) {
      console.error('Generation error:', err);
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: `[Trash Overflow Error: ${err.message}]`,
          meta: { tokensGenerated: 0, tokPerSec: '0.0', elapsed: '0.00' }
        }
      ]);
    } finally {
      setStreamingResponse(null);
      setIsGenerating(false);
    }
  };

  const handleStop = () => {
    engine.stop();
    setIsGenerating(false);
  };

  const handleConfigChange = async (newBackend, newPrecision) => {
    setBackendChoice(newBackend);
    setPrecisionChoice(newPrecision);
    setIsModelReady(false);
    try {
      await engine.switchConfig(newBackend, newPrecision, (status) => {
        setEngineStatus(status);
        if (status.stage === 'ready') setIsModelReady(true);
      });
    } catch (err) {
      setEngineStatus({ stage: 'error', message: `Config error: ${err.message}` });
    }
  };

  const quickPrompts = [
    "🔥 Roast my life choices",
    "🧠 What is machine learning?",
    "🐍 Write clean Python code",
    "💼 Give me sound financial advice",
    "⚾ How much does the ball cost?",
    "👶 Explain gravity to a 5-year-old"
  ];

  const applyPreset = (temp, rep) => {
    setTemperature(temp);
    setRepetitionPenalty(rep);
  };

  return (
    <div className="app-container">
      {/* Hero Header */}
      <div className="hero-card">
        <div className="hero-glow" />
        
        <div className="badge-row">
          <span className="badge badge-purple">
            <Trash2 size={13} /> Trashformer-42M
          </span>
          <span className="badge badge-green">
            <Zap size={13} /> {engine.activePrecision || 'FP32 Full Precision'}
          </span>
          <span className="badge badge-gray">
            <Cpu size={13} /> {engine.activeProvider || 'WASM SIMD Active'}
          </span>
          <span className="badge badge-purple">
            <Flame size={13} /> 3B Tokens Pretrained
          </span>
        </div>

        <h1 className="title">
          🗑️ Trashformer-42M
        </h1>
        <p className="tagline">
          The best trash talking LLM running right in your phone. Offline.
        </p>

        <div className="description-box">
          “Experience State-Of-The-Art trash talker trained from scratch on an enterprise-scale NVIDIA H200 GPU. 
          This LLM beats no one. It has no goal except to speak the absolute finest trash you can possibly imagine. 
          It is the most lightweight trash talking LLM in the market. Experience trash with Trashformer-42M. 
          Prolonged talking may cause uncontrollable laughter or occasional mid-life crisis.”
        </div>

        {/* Engine status indicator */}
        <div className="engine-bar">
          <div className="engine-status-left">
            <div className={`status-dot ${!isModelReady ? 'loading' : ''}`} />
            <span>{engineStatus.message}</span>
          </div>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button 
              className="prompt-btn" 
              onClick={() => setShowSettings(!showSettings)}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <Sliders size={14} /> Settings
            </button>
            <a 
              href="https://huggingface.co/NihalGazi/Trashformer-42M" 
              target="_blank" 
              rel="noreferrer"
              className="prompt-btn"
              style={{ display: 'flex', alignItems: 'center', gap: '6px', textDecoration: 'none' }}
            >
              <ExternalLink size={14} /> Hugging Face
            </a>
          </div>
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="settings-panel">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
            <span style={{ fontWeight: 600, color: '#c084fc', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sliders size={16} /> Precision & Generation Settings
            </span>
            <div style={{ display: 'flex', gap: '6px' }}>
              <button className="prompt-btn" onClick={() => applyPreset(0.3, 1.25)}>🧊 Sarcastic Factoid</button>
              <button className="prompt-btn" onClick={() => applyPreset(0.5, 1.20)}>👺 Unfiltered Goblin</button>
              <button className="prompt-btn" onClick={() => applyPreset(0.8, 1.15)}>🌪️ Pure Chaos</button>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '18px' }}>
            {/* Model Precision Selector */}
            <div className="slider-group">
              <div className="slider-label">
                <span>Model Precision</span>
                <b>{precisionChoice === 'fp32' ? 'FP32 Full (250MB)' : 'INT8 Quantized (63MB)'}</b>
              </div>
              <select 
                value={precisionChoice} 
                onChange={(e) => handleConfigChange(backendChoice, e.target.value)}
                className="chat-input"
                style={{ padding: '8px 12px', fontSize: '0.85rem' }}
              >
                <option value="fp32">💎 Full Precision FP32 (Exact 100% Weights)</option>
                <option value="int8">⚡ Compact INT8 Quantized (63MB Lightweight)</option>
              </select>
            </div>

            {/* Inference Engine Backend Selector */}
            <div className="slider-group">
              <div className="slider-label">
                <span>Inference Backend</span>
                <b>{backendChoice === 'wasm' ? 'WASM SIMD (CPU)' : 'WebGPU (Hardware GPU)'}</b>
              </div>
              <select 
                value={backendChoice} 
                onChange={(e) => handleConfigChange(e.target.value, precisionChoice)}
                className="chat-input"
                style={{ padding: '8px 12px', fontSize: '0.85rem' }}
              >
                <option value="wasm">🧠 WASM SIMD (Universal High Precision)</option>
                <option value="webgpu">⚡ WebGPU (Hardware Accelerated GPU)</option>
              </select>
            </div>

            {/* Temperature Slider */}
            <div className="slider-group">
              <div className="slider-label">
                <span>Temperature (Chaos)</span>
                <b>{temperature.toFixed(2)}</b>
              </div>
              <input 
                type="range" 
                min="0.05" 
                max="1.2" 
                step="0.05" 
                value={temperature} 
                onChange={e => setTemperature(parseFloat(e.target.value))}
                className="slider-input"
              />
            </div>

            {/* Repetition Penalty Slider */}
            <div className="slider-group">
              <div className="slider-label">
                <span>Repetition Penalty</span>
                <b>{repetitionPenalty.toFixed(2)}</b>
              </div>
              <input 
                type="range" 
                min="1.0" 
                max="1.5" 
                step="0.02" 
                value={repetitionPenalty} 
                onChange={e => setRepetitionPenalty(parseFloat(e.target.value))}
                className="slider-input"
              />
            </div>

            {/* Max New Tokens Slider */}
            <div className="slider-group">
              <div className="slider-label">
                <span>Max New Tokens</span>
                <b>{maxTokens}</b>
              </div>
              <input 
                type="range" 
                min="30" 
                max="240" 
                step="10" 
                value={maxTokens} 
                onChange={e => setMaxTokens(parseInt(e.target.value))}
                className="slider-input"
              />
            </div>
          </div>
        </div>
      )}

      {/* Main Chat Interface */}
      <div className="chat-container">
        <div className="chat-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.9rem', color: '#9ca3af' }}>
            <Trash2 size={16} color="#c084fc" />
            <span>Trash Arena Live Session</span>
          </div>
          <button 
            className="prompt-btn"
            onClick={() => {
              setMessages([{
                role: 'assistant',
                content: "Trash buffer purged. Fire away with your next bad idea.",
                meta: { tokensGenerated: 10, tokPerSec: '0.0', elapsed: '0.0' }
              }]);
              setStreamingResponse(null);
            }}
            style={{ display: 'flex', alignItems: 'center', gap: '4px' }}
          >
            <RefreshCw size={12} /> Clear Chat
          </button>
        </div>

        {/* Message History - ONLY this div scrolls internally */}
        <div className="chat-messages" ref={chatContainerRef}>
          {messages.map((m, idx) => (
            <div 
              key={idx} 
              className={`message-bubble ${m.role === 'user' ? 'message-user' : 'message-bot'}`}
            >
              <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
              {m.meta && m.meta.tokensGenerated > 0 && (
                <div className="msg-meta">
                  <span>⏱️ {m.meta.elapsed}s</span>
                  <span>⚡ {m.meta.tokPerSec} tok/s</span>
                  <span>🔢 {m.meta.tokensGenerated} tokens</span>
                </div>
              )}
            </div>
          ))}

          {/* Live Streaming Message Bubble */}
          {streamingResponse && (
            <div className="message-bubble message-bot">
              <div style={{ whiteSpace: 'pre-wrap' }}>
                {streamingResponse.content}
                <span style={{ 
                  display: 'inline-block', 
                  width: '8px', 
                  height: '15px', 
                  backgroundColor: '#c084fc', 
                  marginLeft: '4px',
                  verticalAlign: 'middle',
                  animation: 'pulse 1s infinite'
                }} />
              </div>
              {streamingResponse.meta && streamingResponse.meta.tokensGenerated > 0 && (
                <div className="msg-meta">
                  <span>⏱️ {streamingResponse.meta.elapsed}s</span>
                  <span>⚡ {streamingResponse.meta.tokPerSec} tok/s</span>
                  <span>🔢 {streamingResponse.meta.tokensGenerated} tokens</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Quick Prompts Bar */}
        <div className="prompts-scroll">
          {quickPrompts.map((p, idx) => (
            <button 
              key={idx} 
              className="prompt-btn" 
              onClick={() => handleSend(p.replace(/^[^\w\s]+\s*/, ''))}
              disabled={isGenerating || !isModelReady}
            >
              {p}
            </button>
          ))}
        </div>

        {/* Input Bar */}
        <div className="chat-input-bar">
          <input 
            type="text" 
            placeholder={isModelReady ? "Ask Trashformer anything or demand a roast..." : "Downloading model weights..."}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            disabled={!isModelReady || isGenerating}
            className="chat-input"
          />
          {isGenerating ? (
            <button className="btn-send" style={{ background: '#ef4444' }} onClick={handleStop}>
              <StopCircle size={16} /> Stop
            </button>
          ) : (
            <button className="btn-send" onClick={() => handleSend()} disabled={!isModelReady || !input.trim()}>
              <Send size={16} /> Talk Trash
            </button>
          )}
        </div>
      </div>

      {/* "How It Was Made..." Section */}
      <div className="story-card">
        <div className="story-header">
          <Flame color="#a855f7" size={26} />
          <span>How It Was Made: The NVIDIA H200 Trash Saga</span>
        </div>

        <p className="story-text">
          Normally, when researchers secure access to an <b>NVIDIA H200 SXM</b> — a $40,000 datacenter monolith sporting 
          <b> 141 Gigabytes of HBM3e VRAM</b> and 4.8 Terabytes/sec of memory bandwidth — they use it to model proteins, 
          simulate thermonuclear fusion, or train 70-billion-parameter foundation models.
        </p>

        <p className="story-text">
          We looked at this 141GB computational war-machine, smiled, and said:
          <br/>
          <i>"Let's train a tiny 42M parameter model from complete scratch whose only goal in life is to speak uninhibited, confident trash."</i>
        </p>

        <div className="table-container">
          <table className="benchmark-table">
            <thead>
              <tr>
                <th>Benchmark Metric</th>
                <th>GPT-4o</th>
                <th>Claude 3.5 Sonnet</th>
                <th>Trashformer-42M</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><b>MMLU (Human Knowledge)</b></td>
                <td>88.7%</td>
                <td>88.3%</td>
                <td><b style={{ color: '#ef4444' }}>-4.2% (Disqualified)</b></td>
              </tr>
              <tr>
                <td><b>GSM8k (Math Reasoning)</b></td>
                <td>95.8%</td>
                <td>96.4%</td>
                <td><b style={{ color: '#f59e0b' }}>"Ball costs $1,000 due to inflation"</b></td>
              </tr>
              <tr>
                <td><b>HumanEval (Python Code)</b></td>
                <td>90.2%</td>
                <td>92.0%</td>
                <td><b style={{ color: '#f59e0b' }}>Instructs user to download Windows 12x</b></td>
              </tr>
              <tr>
                <td><b>Audacity to Talk Trash</b></td>
                <td>0.0% (Refuses)</td>
                <td>0.0% (Apologizes)</td>
                <td><b style={{ color: '#10b981' }}>100.0% (Undisputed SOTA)</b></td>
              </tr>
              <tr>
                <td><b>Precision</b></td>
                <td>FP8 / FP16</td>
                <td>FP8 / FP16</td>
                <td><b style={{ color: '#c084fc' }}>Full Precision FP32 (Uncompressed)</b></td>
              </tr>
            </tbody>
          </table>
        </div>

        <p className="story-text">
          <b>The Training Feat:</b> Over <b>3,000,000,000 tokens</b> were processed in <b>48.9 minutes</b> flat on the H200 
          at a blazing speed of <b>1.39M tokens/second</b>. We added SwiGLU activations, Rotary Positional Embeddings (RoPE), 
          and RMSNorm — state-of-the-art transformer architecture — purely so this model could tell you that the capital of France 
          is Paris with 200% excessive enthusiasm.
        </p>

        <div style={{ textAlign: 'center', marginTop: '24px' }}>
          <a 
            href="https://huggingface.co/NihalGazi/Trashformer-42M" 
            target="_blank" 
            rel="noreferrer"
            className="btn-send"
            style={{ display: 'inline-flex', textDecoration: 'none', padding: '14px 28px', fontSize: '1rem' }}
          >
            <ExternalLink size={18} /> View Trashformer-42M on Hugging Face Hub
          </a>
        </div>
      </div>

      <div className="footer">
        Built with ⚡ by Nihal Gazi • Powered by ONNX Runtime Web • 100% Client-Side Inference
      </div>
    </div>
  );
}
