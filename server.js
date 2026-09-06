// server.js — Robust Hybrid OpenAI ↔ NIM Proxy
// Express 5 Compatible
// Fixes: auth bypass, startup DDoS, silent stream failures, memory leaks, Express 5 deprecations

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuration ───────────────────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';
const DISCORD_WEBHOOK_URL = process.env.DISCORD_WEBHOOK_URL;

const MAX_TOKENS_LIMIT = 65536;
const REQUEST_TIMEOUT_MS = 180000;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');
if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');

// ─── Config validation ──────────────────────────────────────────────────────

function validateConfig() {
  const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };
  
  if (!NIM_API_KEY) fatal('NIM_API_KEY is required. Get one at https://build.nvidia.com/');
  
  if (!CLIENT_AUTH_KEY) {
    console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.');
  }
}

validateConfig();

// ─── Model Mapping ─────────────────────────────────────────────────────────

const MODEL_MAPPING = {
  // ── High-Capability & Frontier Chat / RP ──
  'gpt-4-turbo': 'moonshotai/kimi-k3',
  'kimi-k3': 'moonshotai/kimi-k3',
  'moonshotai/kimi-k3': 'moonshotai/kimi-k3',

  'gpt-4o': 'deepseek-ai/deepseek-v4-pro-0813',
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro-0813',
  'deepseek-v4-pro-0813': 'deepseek-ai/deepseek-v4-pro-0813',
  'deepseek-ai/deepseek-v4-pro-0813': 'deepseek-ai/deepseek-v4-pro-0813',

  'gpt-4-flash': 'deepseek-ai/deepseek-v4-flash-0731',
  'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash-0731',
  'deepseek-v4-flash-0731': 'deepseek-ai/deepseek-v4-flash-0731',
  'deepseek-ai/deepseek-v4-flash-0731': 'deepseek-ai/deepseek-v4-flash-0731',

  'gpt-4': 'nvidia/nemotron-3-ultra-550b-a55b',
  'nemotron-ultra': 'nvidia/nemotron-3-ultra-550b-a55b',
  'nvidia/nemotron-3-ultra-550b-a55b': 'nvidia/nemotron-3-ultra-550b-a55b',

  'gpt-3.5-turbo': 'nvidia/nemotron-3-super-120b-a12b',
  'nemotron-super': 'nvidia/nemotron-3-super-120b-a12b',
  'nvidia/nemotron-3-super-120b-a12b': 'nvidia/nemotron-3-super-120b-a12b',

  'gpt-3.5o': 'nvidia/nemotron-mini-4b-instruct',
  'nvidia/nemotron-mini-4b-instruct': 'nvidia/nemotron-mini-4b-instruct',

  'claude-3-opus': 'openai/gpt-oss-120b',
  'openai/gpt-oss-120b': 'openai/gpt-oss-120b',

  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gpt-oss-20b': 'openai/gpt-oss-20b',
  'openai/gpt-oss-20b': 'openai/gpt-oss-20b',

  'gemini-pro': 'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'nvidia/llama-3.3-nemotron-super-49b-v1.5': 'nvidia/llama-3.3-nemotron-super-49b-v1.5',

  'gemini-turbo': 'meta/llama-3.3-70b-instruct',
  'meta/llama-3.3-70b-instruct': 'meta/llama-3.3-70b-instruct',

  'gemini-turbo?': 'abacusai/dracarys-llama-3.1-70b-instruct',
  'abacusai/dracarys-llama-3.1-70b-instruct': 'abacusai/dracarys-llama-3.1-70b-instruct',

  'gpt-3.5': 'qwen/qwen3.5-397b-a17b',
  'qwen/qwen3.5-397b-a17b': 'qwen/qwen3.5-397b-a17b',

  'glm-5.2': 'z-ai/glm-5.2',
  'z-ai/glm-5.2': 'z-ai/glm-5.2',

  'step-3.5-flash': 'stepfun-ai/step-3.5-flash',
  'step-3.7-flash': 'stepfun-ai/step-3.7-flash',

  // ── Reasoning & Agentic Models ──
  'nemotron-lightning': 'nvidia/nemotron-3.5-lightning-30b-a3b',
  'nemotron-3.5-lightning': 'nvidia/nemotron-3.5-lightning-30b-a3b',
  'nvidia/nemotron-3.5-lightning-30b-a3b': 'nvidia/nemotron-3.5-lightning-30b-a3b',

  'muse-glimmer': 'meta/muse-glimmer-30b',
  'muse-glimmer-30b': 'meta/muse-glimmer-30b',
  'meta/muse-glimmer-30b': 'meta/muse-glimmer-30b',

  'laguna-xs': 'poolside/laguna-xs-2.1',
  'laguna-xs-2.1': 'poolside/laguna-xs-2.1',
  'poolside/laguna-xs-2.1': 'poolside/laguna-xs-2.1',

  'diffusiongemma': 'google/diffusiongemma-26b-a4b-it',
  'diffusiongemma-26b': 'google/diffusiongemma-26b-a4b-it',
  'google/diffusiongemma-26b-a4b-it': 'google/diffusiongemma-26b-a4b-it',

  'nemotron-omni': 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
  'nemotron-nano-omni': 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
  'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning': 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',

  // ── Vision & Multimodal Models ──
  'cosmos-reasoner': 'nvidia/cosmos3-nano-reasoner',
  'cosmos3-nano-reasoner': 'nvidia/cosmos3-nano-reasoner',
  'nvidia/cosmos3-nano-reasoner': 'nvidia/cosmos3-nano-reasoner',

  'llama-3.2-11b-vision': 'meta/llama-3.2-11b-vision-instruct',
  'meta/llama-3.2-11b-vision-instruct': 'meta/llama-3.2-11b-vision-instruct',

  'llama-3.2-90b-vision': 'meta/llama-3.2-90b-vision-instruct',
  'meta/llama-3.2-90b-vision-instruct': 'meta/llama-3.2-90b-vision-instruct',

  'paligemma': 'google/google-paligemma',
  'google-paligemma': 'google/google-paligemma',
  'google/google-paligemma': 'google/google-paligemma',

  // ── Mistral Models ──
  'mistral': 'mistralai/mistral-large-3-675b-instruct-2512',
  'mistral-turbo': 'mistralai/mistral-medium-3.5-128b',
  'mistral-pro': 'mistralai/mistral-small-4-119b-2603',
  'mistral-nemo': 'mistralai/mistral-nemotron',
  'mistralai/mistral-nemotron': 'mistralai/mistral-nemotron',
  'mistral-fast': 'mistralai/ministral-14b-instruct-2512',

  // ── Google Gemma Series ──
  'google-light': 'google/gemma-4-31b-it',
  'gemma-4-31b': 'google/gemma-4-31b-it',
  'google/gemma-4-31b-it': 'google/gemma-4-31b-it',
  'google-lighter': 'google/gemma-3n-e4b-it',
  'google-lightest': 'google/gemma-2-2b-it',

  // ── Specialized: Translation & Voicechat ──
  'nemotron-voicechat': 'nvidia/nemotron-voicechat',
  'nvidia/nemotron-voicechat': 'nvidia/nemotron-voicechat',

  'riva-translate': 'nvidia/riva-translate-4b-instruct-v2',
  'riva-translate-v2': 'nvidia/riva-translate-4b-instruct-v2',
  'nvidia/riva-translate-4b-instruct-v2': 'nvidia/riva-translate-4b-instruct-v2',
  'riva-translate-v1': 'nvidia/riva-translate-4b-instruct-v1_1',
  'nvidia/riva-translate-4b-instruct-v1_1': 'nvidia/riva-translate-4b-instruct-v1_1',

  // ── Specialized: Calibration VLMs ──
  'ising-1.5': 'nvidia/ising-calibration-1.5-31b',
  'nvidia/ising-calibration-1.5-31b': 'nvidia/ising-calibration-1.5-31b',
  'ising-1': 'nvidia/ising-calibration-1-35b-a3b',
  'nvidia/ising-calibration-1-35b-a3b': 'nvidia/ising-calibration-1-35b-a3b',

  // ── Active MiniMax Models (Deprecation Notice in Catalog) ──
  // Kept active per user preference; nvidia marked minimax-m3 as "Deprecation in 3d"
  'm3': 'minimaxai/minimax-m3',
  'minimax-m3': 'minimaxai/minimax-m3',
  'minimaxai/minimax-m3': 'minimaxai/minimax-m3',
  'm2.7': 'minimaxai/minimax-m2.7',
  'minimax-m2.7': 'minimaxai/minimax-m2.7'
};

/*
 ─── FREE NIM CATALOG: NON-CHAT / SPECIALIZED USE-CASE MODELS ────────────────
 The following models are free endpoints on build.nvidia.com, but they do NOT
 expose an OpenAI-compatible /v1/chat/completions endpoint.
 They are documented here for catalog familiarity, reference, and manual API calls:

 [1. AUDIO PROCESSING & TTS]
 • 'nvidia/bnr'
   - Name: Background Noise Removal
   - Use Case: Cleans unwanted ambient noise from audio to improve speech intelligibility.
   - Endpoint: Specialized Audio REST endpoint.

 • 'nvidia/studiovoice'
   - Name: Studio Voice
   - Use Case: Enhances noisy, low-quality microphone speech into studio-quality audio.
   - Endpoint: Specialized Audio REST endpoint.

 • 'nvidia/magpie-tts-zeroshot'
   - Name: Magpie TTS Zero-Shot
   - Use Case: Expressive and engaging text-to-speech generated from short audio samples.
   - Endpoint: Dedicated Text-to-Speech (TTS) endpoint.

 [2. VIDEO GENERATION & WORLD MODELS]
 • 'nvidia/cosmos3-nano'
   - Name: Cosmos 3 Nano
   - Use Case: Generates physics-aware videos from text or image prompts for physical AI development.
   - Endpoint: Specialized Video Generation endpoint.

 • 'nvidia/cosmos-transfer2.5-2b'
   - Name: Cosmos Transfer 2.5 2B
   - Use Case: Generates physics-aware video world states using spatial control & simulation inputs.
   - Endpoint: Synthetic Data Generation / World Model endpoint.

 • 'nvidia/cosmos-transfer1-7b'
   - Name: Cosmos Transfer 1 7B (Deprecation in 3d)
   - Use Case: Preceding generation video world state model.
   - Endpoint: Synthetic Data Generation endpoint.

 [3. COMPUTER VISION & VIDEO ANALYSIS]
 • 'nvidia/synthetic-video-detector'
   - Name: Synthetic Video Detector
   - Use Case: AI microservice to detect whether video content is AI-generated (deepfake detection).
   - Endpoint: Specialized Computer Vision endpoint.

 • 'nvidia/active-speaker-detection'
   - Name: Active Speaker Detection
   - Use Case: Identifies and tracks speaker bounding boxes across sequential video frames.
   - Endpoint: Specialized Computer Vision endpoint.

 [4. AUTONOMOUS DRIVING & 3D PERCEPTION]
 • 'nvidia/streampetr'
   - Name: StreamPETR
   - Use Case: Multi-frame temporal object detection using sparse queries for autonomous driving.
   - Endpoint: Robotics / Autonomous Vehicle perception endpoint.

 • 'nvidia/sparsedrive'
   - Name: SparseDrive
   - Use Case: End-to-end autonomous driving stack: perception, prediction, and planning.
   - Endpoint: Robotics / Autonomous Vehicle perception endpoint.

 • 'nvidia/bevformer'
   - Name: BEVFormer
   - Use Case: Multi-camera Bird's-Eye-View (BEV) 3D perception transformer for vehicles.
   - Endpoint: Robotics / Autonomous Vehicle perception endpoint.

 [5. EMBEDDINGS & RETRIEVAL]
 • 'nvidia/nemotron-3-embed-1b'
   - Name: Nemotron 3 Embed 1B
   - Use Case: High-efficiency 1B embedding model for RAG, vector databases, and semantic search.
   - Endpoint: Uses /v1/embeddings (OpenAI format embedding endpoint).

 [6. CONTENT MODERATION & SAFETY CLASSIFIERS]
 • 'nvidia/nemotron-3.5-content-safety'
   - Name: Nemotron 3.5 Content Safety
   - Use Case: Multilingual & multimodal moderation classifier for toxic/unsafe content.
   - Endpoint: Classification / Guardrails endpoint.

 • 'nvidia/llama-3_1-nemotron-safety-guard-8b-v3'
   - Name: Llama 3.1 Nemotron Safety Guard 8B v3
   - Use Case: Moderation guard model for scanning inputs/outputs against safety taxonomy.
   - Endpoint: Classification / Guardrails endpoint.

 • 'meta/llama-guard-4-12b'
   - Name: Llama Guard 4 12B
   - Use Case: Multimodal input/output safety classifier from Meta.
   - Endpoint: Classification / Guardrails endpoint.
───────────────────────────────────────────────────────────────────────────────
*/

const FALLBACK_MODELS = [
  'mistralai/mistral-medium-3.5-128b',
  'mistralai/mistral-small-4-119b-2603',
  'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'google/gemma-4-31b-it'
];

// ─── Middleware ─────────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// FIX: Extract token AFTER "Bearer " prefix, compare only the token
// Prevents bypass when CLIENT_AUTH_KEY is empty (expected would be "Bearer " which is 7 chars)
function extractBearerToken(authHeader) {
  if (!authHeader || typeof authHeader !== 'string') return null;
  const parts = authHeader.trim().split(' ');
  if (parts.length !== 2 || parts[0] !== 'Bearer') return null;
  return parts[1];
}

function safeTimingEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  try {
    return timingSafeEqual(Buffer.from(a), Buffer.from(b));
  } catch {
    return false;
  }
}

app.use((req, res, next) => {
  if (req.path === '/health' || req.path === '/v1/models') {
    return next();
  }

  const token = extractBearerToken(req.headers.authorization);
  
  if (!token || !CLIENT_AUTH_KEY) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid or missing authentication',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  if (!safeTimingEqual(token, CLIENT_AUTH_KEY)) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid authentication credentials',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  next();
});

// ─── Validation ─────────────────────────────────────────────────────────────

// FIX: Use lightweight model listing instead of burning inference quota
// If NIM doesn't support /models, skip validation entirely rather than DDoS-ing yourself
async function validateModels() {
  if (SKIP_VALIDATION) {
    console.log('[VALIDATION] Skipped (SKIP_VALIDATION=true)');
    return;
  }

  console.log('[VALIDATION] Checking model availability via /v1/models...');

  try {
    const response = await axios.get(`${NIM_API_BASE}/models`, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: VALIDATION_TIMEOUT_MS
    });

    const availableModels = new Set(
      (response.data.data || []).map(m => m.id)
    );

    const invalid = [];
    const checked = new Set();
    
    for (const [alias, nimId] of Object.entries(MODEL_MAPPING)) {
      if (checked.has(nimId)) continue;
      checked.add(nimId);

      if (availableModels.has(nimId)) {
        console.log(`[VALIDATION] ✓ ${alias} → ${nimId}`);
      } else {
        console.warn(`[VALIDATION] ✗ ${alias} → ${nimId} (not in catalog)`);
        invalid.push({ alias, nimId, error: 'Model not found in NIM catalog' });
      }
    }

    if (invalid.length > 0) {
      await sendDiscordAlert(invalid);
    } else {
      console.log('[VALIDATION] All models valid.');
    }

  } catch (err) {
    console.warn(`[VALIDATION] /v1/models endpoint failed: ${err.message}. Skipping validation.`);
    console.warn('[VALIDATION] Consider setting SKIP_VALIDATION=true if your NIM provider lacks a model listing endpoint.');
  }
}

async function sendDiscordAlert(invalidModels) {
  if (!DISCORD_WEBHOOK_URL) return;

  const embed = {
    title: '⚠️ NIM Proxy: Model Validation Failed',
    description: `${invalidModels.length} model(s) failed validation. Check NIM catalog for deprecations.`,
    color: 0xff4444,
    timestamp: new Date().toISOString(),
    fields: invalidModels.map(m => ({
      name: `\`${m.alias}\``,
      value: `Backend: \`${m.nimId}\`\nError: \`${m.error}\``,
      inline: true
    }))
  };

  try {
    await axios.post(DISCORD_WEBHOOK_URL, {
      embeds: [embed],
      username: 'NIM Proxy Monitor'
    }, { timeout: 5000 });
    console.log('[DISCORD] Alert sent.');
  } catch (err) {
    console.error('[DISCORD] Failed to send alert:', err.message);
  }
}

// ─── Helper: Safe Stream Writing ───────────────────────────────────────────

// FIX: Wrap res.write in try/catch to prevent crashes on closed sockets
function safeWrite(res, data) {
  try {
    if (!res.writableEnded && !res.destroyed && res.writable) {
      res.write(data);
      return true;
    }
  } catch (err) {
    console.warn('[STREAM] Write failed:', err.message);
  }
  return false;
}

// ─── Helper: Fallback Chain ─────────────────────────────────────────────────

async function callWithFallback(baseRequest, models) {
  let lastError = null;

  for (const model of models) {
    try {
      const res = await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        { ...baseRequest, model },
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: baseRequest.stream ? 'stream' : 'json',
          timeout: REQUEST_TIMEOUT_MS
        }
      );

      return { response: res, model };

    } catch (err) {
      lastError = err;
      console.warn(
        `[FALLBACK] Model failed: ${model}`,
        err.response?.status,
        err.response?.data?.error?.message || err.message
      );
    }
  }

  throw lastError || new Error('All models failed');
}

// ─── Routes ────────────────────────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '2.2.0' });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nim-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  let streamEndedCleanly = false;
  let upstreamStream = null;

  try {
    const {
      model,
      messages,
      temperature,
      max_tokens,
      stream
    } = req.body;

    // Passthrough support: Use alias if found, otherwise pass model directly if provided
    const primaryModel = MODEL_MAPPING[model] || model || 'nvidia/llama-3.3-nemotron-super-49b-v1.5';
    const modelChain = [primaryModel, ...FALLBACK_MODELS.filter(m => m !== primaryModel)];

    const baseRequest = {
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? 2048, MAX_TOKENS_LIMIT),
      stream: stream || false,
      extra_body: ENABLE_THINKING_MODE
        ? { chat_template_kwargs: { thinking: true } }
        : undefined
    };

    const { response, model: usedModel } = await callWithFallback(baseRequest, modelChain);
    upstreamStream = response.data;
    console.log('[PROXY] Model used:', usedModel);

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const decoder = new StringDecoder('utf8');
      let buffer = '';
      let reasoningOpen = false;
      let doneSent = false;
      let cleanedUp = false;

      const cleanup = () => {
        if (cleanedUp) return;
        cleanedUp = true;
        if (upstreamStream) {
          upstreamStream.removeAllListeners();
        }
        req.removeAllListeners('close');
      };

      const processLine = (line) => {
        if (!line.startsWith('data: ')) return;

        if (line.includes('[DONE]')) {
          if (!doneSent) {
            safeWrite(res, 'data: [DONE]\n\n');
            doneSent = true;
          }
          streamEndedCleanly = true;
          return;
        }

        try {
          const data = JSON.parse(line.slice(6));
          const delta = data.choices?.[0]?.delta;

          if (delta) {
            let content = delta.content || '';
            const reasoning = delta.reasoning_content;

            if (SHOW_REASONING) {
              if (reasoning && !reasoningOpen) {
                content = `<thinking>\n${reasoning.replace(/\n/g, '\\n')}`;
                reasoningOpen = true;
              } else if (reasoning) {
                content = reasoning.replace(/\n/g, '\\n');
              }

              if (delta.content && reasoningOpen) {
                content += `\n</thinking>\n\n${delta.content}`;
                reasoningOpen = false;
              }
            }

            delta.content = content;
            delete delta.reasoning_content;
          }

          safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);

        } catch (parseErr) {
          // FIX: Don't silently swallow—send error to client so they know data was lost
          console.warn('[STREAM] Invalid JSON line:', line.slice(0, 100));
          safeWrite(res, `data: ${JSON.stringify({ 
            error: { 
              message: 'Upstream sent malformed chunk', 
              type: 'stream_parse_error',
              details: line.slice(0, 100)
            } 
          })}\n\n`);
        }
      };

      upstreamStream.on('data', chunk => {
        buffer += decoder.write(chunk);

        if (buffer.length > MAX_BUFFER_SIZE) {
          console.error('[STREAM] Buffer overflow, destroying connection');
          safeWrite(res, `data: ${JSON.stringify({ 
            error: { 
              message: 'Stream buffer overflow', 
              type: 'stream_error' 
            } 
          })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
          upstreamStream.destroy();
          cleanup();
          return;
        }

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          processLine(line);
        }
      });

      upstreamStream.on('end', () => {
        buffer += decoder.end();

        if (buffer.trim()) {
          for (const line of buffer.split('\n')) {
            processLine(line);
          }
        }

        if (!doneSent) {
          safeWrite(res, 'data: [DONE]\n\n');
        }

        streamEndedCleanly = true;
        if (!res.writableEnded) {
          res.end();
        }
        cleanup();
      });

      upstreamStream.on('error', err => {
        console.error('[STREAM] Upstream error:', err.message);
        
        if (!res.writableEnded) {
          safeWrite(res, `data: ${JSON.stringify({
            error: {
              message: 'Stream interrupted by upstream error',
              type: 'stream_error'
            }
          })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
        }
        cleanup();
      });

      // FIX: Check req.destroyed (Node/Express 5) 
      // Don't destroy already-finished streams
      req.on('close', () => {
        const clientGone = req.destroyed || !res.writable;
        
        if (!streamEndedCleanly && clientGone) {
          console.warn('[STREAM] Client disconnected prematurely');
        }

        if (upstreamStream && !upstreamStream.destroyed && !streamEndedCleanly) {
          upstreamStream.destroy();
        }
        cleanup();
      });

    } else {
      // Non-streaming response
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: (response.data.choices || []).map((choice, i) => {
          let content = choice.message?.content || '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            const safeReasoning = choice.message.reasoning_content.replace(/\n/g, '\\n');
            content = `<thinking>\n${safeReasoning}\n</thinking>\n\n${content}`;
          }

          return {
            index: i,
            message: {
              role: choice.message?.role || 'assistant',
              content,
              tool_calls: choice.message?.tool_calls
            },
            finish_reason: choice.finish_reason || 'stop'
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('[PROXY] Fatal error:', error.message);
    console.error('[PROXY] NIM response:', error.response?.data);

    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message: error.message,
          type: 'invalid_request_error',
          code: error.response?.status || 500
        }
      });
    } else if (!res.writableEnded) {
      safeWrite(res, `data: ${JSON.stringify({
        error: {
          message: error.message,
          type: 'proxy_error'
        }
      })}\n\n`);
      safeWrite(res, 'data: [DONE]\n\n');
      res.end();
    }

    // Clean up upstream stream if we have it
    if (upstreamStream && !upstreamStream.destroyed) {
      upstreamStream.destroy();
    }
  }
});

// FIX: Express 5 named wildcard — but use proper 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.method} ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// ─── Startup ───────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`[PROXY] Hybrid proxy running on port ${PORT}`);
  console.log(`[PROXY] Max tokens limit: ${MAX_TOKENS_LIMIT}`);
  
  // Run validation after server starts, non-blocking
  validateModels().catch(err => {
    console.error('[VALIDATION] Startup check failed:', err.message);
  });
});
