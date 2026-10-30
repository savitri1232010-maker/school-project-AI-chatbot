import express from 'express';
import { createServer as createViteServer } from 'vite';
import {
  GoogleGenAI,
  Type,
  ThinkingLevel,
  GenerateContentParameters,
  GenerateContentResponse,
} from '@google/genai';
import dotenv from 'dotenv';
import path from 'path';
import fs from 'fs';

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: '25mb' }));

function getGeminiClient() {
  return new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// Ordered from lowest latency (zero thinking overhead) to fallback models
const GEMINI_FAST_MODELS: Array<{
  model: string;
  thinkingLevel?: ThinkingLevel;
}> = [
  { model: 'gemini-3.1-flash-lite', thinkingLevel: ThinkingLevel.MINIMAL },
  { model: 'gemini-3.8-flash', thinkingLevel: ThinkingLevel.LOW },
  { model: 'gemini-flash-latest' },
];

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`Timeout after ${ms}ms on ${label}`));
    }, ms);
    promise
      .then((val) => {
        clearTimeout(timer);
        resolve(val);
      })
      .catch((err) => {
        clearTimeout(timer);
        reject(err);
      });
  });
}

async function generateWithFallback(
  params: Omit<GenerateContentParameters, 'model'>,
  timeoutMs = 9000
): Promise<{ response: GenerateContentResponse; usedModel: string }> {
  const ai = getGeminiClient();
  let lastError: unknown = null;

  for (const candidate of GEMINI_FAST_MODELS) {
    try {
      const mergedConfig: Record<string, unknown> = {
        ...(params.config || {}),
      };
      if (candidate.thinkingLevel) {
        mergedConfig.thinkingConfig = { thinkingLevel: candidate.thinkingLevel };
      }

      const response = await withTimeout(
        ai.models.generateContent({
          ...params,
          model: candidate.model,
          config: mergedConfig,
        }),
        timeoutMs,
        candidate.model
      );

      const textOut = (response.text ?? '').trim();
      if (textOut.length > 0) {
        return { response, usedModel: candidate.model };
      }
      lastError = new Error(`Empty response from ${candidate.model}`);
    } catch (err: unknown) {
      lastError = err;
    }
  }

  throw lastError || new Error('All Gemini models timed out or failed.');
}

function formatCleanError(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error);
  const lower = raw.toLowerCase();
  if (
    lower.includes('429') ||
    lower.includes('quota') ||
    lower.includes('resource_exhausted') ||
    lower.includes('rate limit') ||
    lower.includes('too many requests')
  ) {
    return 'Limit reached. Please switch to Offline mode in the top-right corner to continue using Quickio on your local GPU.';
  }
  if (
    lower.includes('503') ||
    lower.includes('unavailable') ||
    lower.includes('high demand') ||
    lower.includes('timeout')
  ) {
    return 'Quickio Cloud is momentarily busy. Please try again in a few seconds or switch to Offline mode.';
  }
  return 'Limit reached or connection interrupted. Please switch to Offline mode or try again shortly.';
}

// 1A. Real-Time Streaming Mode Router endpoint (SSE) with strict timeout & guaranteed non-empty text
app.post('/api/quickio/stream', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders?.();

  const sendEvent = (payload: Record<string, unknown>) => {
    res.write(`data: ${JSON.stringify(payload)}\n\n`);
  };

  try {
    const {
      prompt,
      imageBase64,
      imageMimeType,
      mode = 'Network',
      systemInstruction,
      ollamaModel = 'gemma3:12b',
      responseStyle = 'balanced',
    } = req.body;

    if (!prompt && !imageBase64) {
      sendEvent({ type: 'error', error: 'Prompt or image payload is required.' });
      res.end();
      return;
    }

    const styleNote =
      responseStyle === 'concise'
        ? ' Keep your response brief, direct, and concise (2 to 4 sentences).'
        : responseStyle === 'detailed'
        ? ' Provide a comprehensive, deeply detailed breakdown with structured headings.'
        : ' Provide a clear, well-structured, and direct answer.';

    if (mode === 'Offline') {
      let offlineText =
        `[Quickio Offline Mode -> http://localhost:11434]\n` +
        `Active Local GPU Model: ${ollamaModel}\n` +
        `Make sure Ollama is running on your Windows PC ('ollama serve' with OLLAMA_ORIGINS="*") or run Quickio_AI.exe locally.`;

      try {
        const ollamaPayload: Record<string, unknown> = {
          model: ollamaModel,
          prompt: systemInstruction
            ? `${systemInstruction}${styleNote}\n\nUser Input: ${prompt}`
            : prompt,
          stream: false,
        };
        if (imageBase64) {
          ollamaPayload.images = [imageBase64];
        }

        const localRes = await fetch('http://localhost:11434/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(ollamaPayload),
          signal: AbortSignal.timeout(2500),
        });

        if (localRes.ok) {
          const data = (await localRes.json()) as { response?: string };
          if (data.response && data.response.trim()) {
            offlineText = data.response;
          }
        }
      } catch {
        // Local Ollama daemon not reachable from cloud preview container
      }

      sendEvent({ type: 'chunk', text: offlineText });
      sendEvent({ type: 'done', fullText: offlineText });
      res.end();
      return;
    }

    const parts: Array<{ text?: string; inlineData?: { mimeType: string; data: string } }> = [];

    if (imageBase64 && imageMimeType) {
      parts.push({
        inlineData: {
          mimeType: imageMimeType,
          data: imageBase64,
        },
      });
    }

    parts.push({
      text: prompt || 'Analyze the attached photo clearly and concisely.',
    });

    const sysPromptText =
      (systemInstruction ||
        'You are Quickio AI, a fast multimodal assistant for vision, ecology, safety, and productivity.') +
      styleNote;

    const ai = getGeminiClient();
    let fullText = '';
    let lastError: unknown = null;

    // Try streaming from fastest models first; if a model stalls (>7.5s) or yields empty text, automatically failover
    for (const candidate of GEMINI_FAST_MODELS) {
      try {
        const mergedConfig: Record<string, unknown> = {
          systemInstruction: sysPromptText,
        };
        if (candidate.thinkingLevel) {
          mergedConfig.thinkingConfig = { thinkingLevel: candidate.thinkingLevel };
        }

        const streamPromise = ai.models.generateContentStream({
          model: candidate.model,
          contents: { parts },
          config: mergedConfig,
        });

        const stream = await withTimeout(streamPromise, 7500, `${candidate.model} stream init`);

        const iterator = stream[Symbol.asyncIterator]();
        while (true) {
          const nextResult = await withTimeout(
            iterator.next(),
            7500,
            `${candidate.model} stream chunk`
          );
          if (nextResult.done) break;
          const chunk = nextResult.value as GenerateContentResponse;
          const piece = chunk.text || '';
          if (piece) {
            fullText += piece;
            sendEvent({ type: 'chunk', text: piece });
          }
        }

        if (fullText.trim().length > 0) {
          break;
        }
      } catch (err: unknown) {
        lastError = err;
        if (fullText.trim().length > 0) {
          break;
        }
      }
    }

    // Final safety fallback: if streaming yielded 0 characters, run fast non-streaming generateWithFallback
    if (fullText.trim().length === 0) {
      const { response } = await generateWithFallback(
        {
          contents: { parts },
          config: { systemInstruction: sysPromptText },
        },
        8500
      );
      const fallbackText = (response.text ?? '').trim();
      if (fallbackText) {
        fullText = fallbackText;
        sendEvent({ type: 'chunk', text: fallbackText });
      } else if (lastError) {
        throw lastError;
      }
    }

    sendEvent({ type: 'done', fullText });
    res.end();
  } catch (error: unknown) {
    sendEvent({ type: 'error', error: formatCleanError(error) });
    res.end();
  }
});

// 1B. Centralized Mode Router JSON endpoint mirroring Python's get_ai_response(prompt, image, mode)
app.post('/api/quickio/respond', async (req, res) => {
  try {
    const {
      prompt,
      imageBase64,
      imageMimeType,
      mode = 'Network',
      systemInstruction,
      ollamaModel = 'gemma3:12b',
      responseStyle = 'balanced',
    } = req.body;

    if (!prompt && !imageBase64) {
      res.status(400).json({ error: 'Prompt or image payload is required.' });
      return;
    }

    const styleNote =
      responseStyle === 'concise'
        ? ' Keep your response brief, direct, and concise.'
        : responseStyle === 'detailed'
        ? ' Provide a comprehensive, deeply detailed breakdown with structured headings.'
        : '';

    if (mode === 'Offline') {
      try {
        const ollamaPayload: Record<string, unknown> = {
          model: ollamaModel,
          prompt: systemInstruction
            ? `${systemInstruction}${styleNote}\n\nUser Input: ${prompt}`
            : prompt,
          stream: false,
        };
        if (imageBase64) {
          ollamaPayload.images = [imageBase64];
        }

        const localRes = await fetch('http://localhost:11434/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(ollamaPayload),
          signal: AbortSignal.timeout(2500),
        });

        if (localRes.ok) {
          const data = (await localRes.json()) as { response?: string };
          res.json({
            mode: 'Offline',
            endpoint: 'http://localhost:11434/api/generate',
            model: ollamaModel,
            text: data.response || 'Empty response from local Ollama tunnel.',
          });
          return;
        }
      } catch {
        // Local Ollama daemon not reachable from cloud preview container
      }

      res.json({
        mode: 'Offline',
        endpoint: 'http://localhost:11434/api/generate',
        model: ollamaModel,
        text:
          `[Quickio Offline Mode -> http://localhost:11434]\n` +
          `Model: ${ollamaModel} (Local NVIDIA RTX 4050 Tunnel)\n` +
          `Run Quickio_AI.exe on Windows with 'ollama serve' running to execute offline, or switch to 'Network mode' in the top-right header.`,
      });
      return;
    }

    const parts: Array<{ text?: string; inlineData?: { mimeType: string; data: string } }> = [];

    if (imageBase64 && imageMimeType) {
      parts.push({
        inlineData: {
          mimeType: imageMimeType,
          data: imageBase64,
        },
      });
    }

    parts.push({
      text: prompt || 'Analyze the attached photo in detail.',
    });

    const { response, usedModel } = await generateWithFallback({
      contents: { parts },
      config: {
        systemInstruction:
          (systemInstruction ||
            'You are Quickio AI, a multimodal desktop assistant for vision, ecology, safety, and productivity.') +
          styleNote,
      },
    });

    res.json({
      mode: 'Network',
      endpoint: 'Quickio Cloud API',
      model: usedModel,
      text: response.text || 'Please try your question again.',
    });
  } catch (error: unknown) {
    res.status(500).json({
      error: formatCleanError(error),
    });
  }
});

// 2. Gemini-Style Interactive Bilingual Quiz Generator
app.post('/api/quickio/quiz', async (req, res) => {
  try {
    const { topic = 'General Science & Technology', primaryLang = 'English', secondaryLang = 'Hindi' } = req.body;

    const { response } = await generateWithFallback({
      contents: `Create a 5-question interactive bilingual multiple-choice quiz on the topic: "${topic}".
Provide each question and its 4 options in both ${primaryLang} and ${secondaryLang} (e.g. "${primaryLang} / ${secondaryLang}").
Include a helpful hint and a clear bilingual explanation for why the correct answer is right.`,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              question: { type: Type.STRING },
              options: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
              correctIndex: { type: Type.INTEGER },
              hint: { type: Type.STRING },
              explanation: { type: Type.STRING },
            },
            required: ['question', 'options', 'correctIndex', 'hint', 'explanation'],
          },
        },
      },
    });

    const questions = JSON.parse((response.text ?? '[]').trim());
    res.json({ questions });
  } catch (error: unknown) {
    res.status(500).json({ error: formatCleanError(error), questions: [] });
  }
});

// 3. Plain-Text to Visual Flowchart Generator
app.post('/api/quickio/flowchart', async (req, res) => {
  try {
    const { textPrompt } = req.body;
    if (!textPrompt) {
      res.status(400).json({ error: 'textPrompt is required' });
      return;
    }

    const { response } = await generateWithFallback({
      contents: `Convert the following plain text description or topic into a structured step-by-step flowchart with 5 to 8 clear sequential nodes: "${textPrompt}".
For each node specify nodeType as "start", "process", "decision", or "end", a concise title, a short 1-sentence description, and an optional branchLabel if it is a decision step.`,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            title: { type: Type.STRING },
            summary: { type: Type.STRING },
            nodes: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  stepNumber: { type: Type.INTEGER },
                  nodeType: { type: Type.STRING },
                  title: { type: Type.STRING },
                  description: { type: Type.STRING },
                  branchLabel: { type: Type.STRING },
                },
                required: ['id', 'stepNumber', 'nodeType', 'title', 'description'],
              },
            },
          },
          required: ['title', 'summary', 'nodes'],
        },
      },
    });

    const flowchart = JSON.parse((response.text ?? '{}').trim());
    res.json({ flowchart });
  } catch (error: unknown) {
    res.status(500).json({ error: formatCleanError(error) });
  }
});

// 4. Ultra-Fast Multi-Object Vision Bounding Box Detection endpoint for live camera tracking
app.post('/api/quickio/detect-boxes', async (req, res) => {
  try {
    const { imageBase64, imageMimeType = 'image/jpeg' } = req.body;
    if (!imageBase64) {
      res.status(400).json({ error: 'imageBase64 is required' });
      return;
    }

    const { response } = await generateWithFallback(
      {
        contents: {
          parts: [
            {
              inlineData: {
                mimeType: imageMimeType,
                data: imageBase64,
              },
            },
            {
              text: `Detect all distinct foreground objects and people currently visible in this camera frame (up to 12 simultaneous objects, e.g., person, face, hand, eyeglasses, cell phone, bottle, cup, pen, book, notebook, plant, leaf, fruit, watch, keyboard, mouse, headphones, bag, etc.).
Rules:
1. Return separate bounding boxes for EACH distinct foreground object visible (even when 2, 3, or more objects are held up at the same time).
2. Coordinates [ymin, xmin, ymax, xmax] must be normalized 0 to 1000 and tightly fit each object.
3. Do NOT draw boxes around empty walls, ceilings, plain backgrounds, shadows, or room walls. If an object was removed and no foreground objects are present, return [].`,
            },
          ],
        },
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                label: { type: Type.STRING },
                confidence: { type: Type.NUMBER },
                ymin: { type: Type.NUMBER },
                xmin: { type: Type.NUMBER },
                ymax: { type: Type.NUMBER },
                xmax: { type: Type.NUMBER },
              },
              required: ['label', 'confidence', 'ymin', 'xmin', 'ymax', 'xmax'],
            },
          },
        },
      },
      6000
    );

    const raw = (response.text ?? '[]').trim();
    const detections = JSON.parse(raw);
    res.json({ detections: Array.isArray(detections) ? detections : [] });
  } catch (error: unknown) {
    res.status(500).json({ error: formatCleanError(error), detections: [] });
  }
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    if (fs.existsSync(distPath)) {
      app.use(express.static(distPath));
      app.get('*', (_req, res) => {
        res.sendFile(path.join(distPath, 'index.html'));
      });
    }
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Quickio AI Desktop Studio server running on http://localhost:${PORT}`);
  });
}

startServer();
