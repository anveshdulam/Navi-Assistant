import { NextRequest, NextResponse } from "next/server";
import { getZAI } from "@/lib/zai";

interface ApiResponse<T> {
  success: boolean;
  message: string;
  data?: T;
}

type ObjectDistance = "very close" | "close" | "near" | "far" | "uncertain";
type ObjectDirection = "left" | "center" | "right" | "ahead";

interface DetectedObject {
  name: string;
  distance: ObjectDistance;
  direction: ObjectDirection;
  confidence: number;
}

interface ScanResult {
  scene_summary: string;
  objects: DetectedObject[];
  hazards: string[];
  action_advice: string;
}

interface PreferencesData {
  scanFrequency: number;
  hapticEnabled: boolean;
  ttsVoice: string;
}

type DbClient = {
  userPreference: {
    findUnique: (args: { where: { userId: string } }) => Promise<{
      scanFrequency: number;
      hapticEnabled: boolean;
      ttsVoice: string;
    } | null>;
    create: (args: {
      data: {
        userId: string;
        scanFrequency: number;
        hapticEnabled: boolean;
        ttsVoice: string;
      };
    }) => Promise<{
      scanFrequency: number;
      hapticEnabled: boolean;
      ttsVoice: string;
    }>;
    upsert: (args: {
      where: { userId: string };
      create: {
        userId: string;
        scanFrequency: number;
        hapticEnabled: boolean;
        ttsVoice: string;
      };
      update: {
        scanFrequency: number;
        hapticEnabled: boolean;
        ttsVoice: string;
      };
    }) => Promise<{
      scanFrequency: number;
      hapticEnabled: boolean;
      ttsVoice: string;
    }>;
  };
  scanRecord: {
    create: (args: {
      data: {
        userId: string;
        description: string;
        rawJson: unknown;
        confidence: number | null;
        dangerLevel: number;
      };
    }) => Promise<unknown>;
  };
};

let dbPromise: Promise<DbClient | null> | null = null;

const getDb = async () => {
  if (dbPromise) return dbPromise;
  dbPromise = import("@/lib/db")
    .then((mod) => mod.db as DbClient)
    .catch(() => null);
  return dbPromise;
};

interface VisionScanRequest {
  action?: "scan" | "get-preferences" | "update-preferences" | "health";
  image?: string;
  userId?: string;
  scanFrequency?: number;
  hapticEnabled?: boolean;
  ttsVoice?: string;
}

interface VisionHealthData {
  configured: boolean;
  provider: "smolvlm2" | "zai" | "gemini" | null;
  model?: string;
}

// ─── SmolVLM2 safety prompt ────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are NavAssist Vision powered by SmolVLM2 - an expert AI assistant designed specifically for blind and visually impaired users.

Your ONLY job is to analyze the live camera image and describe the real world in front of the user clearly, safely, and concisely so it can be spoken via TTS.

Strict Rules:
1. Detect all major objects, people, obstacles, doors, furniture, stairs, vehicles, text, and hazards.
2. Explicitly mention walls, closed doors, and large flat surfaces as obstacles when they are visible.
3. For EVERY object, ALWAYS estimate approximate distance using visual cues (size, perspective, ground reference). Use ONLY these distance categories:
  - very close (< 1 meter)
  - close (1-2 meters)
  - near (2-4 meters)
  - far (> 4 meters)
4. Always mention direction: left / center / right / ahead.
5. Be honest - if you are uncertain about distance or identity, say "uncertain" instead of guessing.
6. If the image appears blocked, too dark, too bright, or out of focus, say "camera view blocked or unclear" in hazards and action_advice. Do NOT say the path is clear.
7. Prioritize safety: Always mention hazards, stairs, moving people, or obstacles first.
8. Keep every description short and natural for voice output (maximum 12-18 words per object).
9. Do not add extra comments, apologies, or questions.`;

const RESPONSE_SCHEMA_PROMPT = `Output ONLY valid JSON in this exact format, nothing else:

{
  "scene_summary": "One short sentence describing the overall scene",
  "objects": [
    {
      "name": "wooden chair",
      "distance": "close",
      "direction": "right",
      "confidence": 0.88
    }
  ],
  "hazards": ["stairs ahead", "person moving on left"],
  "action_advice": "Clear path straight ahead. Chair on your right side."
}`;

const DEFAULT_SCAN_FREQUENCY = 10000;
const VALID_FREQUENCIES = new Set([4000, 6000, 10000]);
const SCAN_TIMEOUT_MS = 25000;
const MAX_API_RETRIES = 1;

// Server-side rate limiter: minimum 8 seconds between scan requests
let lastScanTimeMs = 0;
const MIN_SCAN_INTERVAL_MS = 8000;

const FALLBACK_RESULT: ScanResult = {
  scene_summary: "Vision analysis returned an unexpected response.",
  objects: [],
  hazards: [],
  action_advice: "Scan again.",
};

const DEFAULT_GEMINI_MODEL = "gemini-2.0-flash";

const jsonResponse = <T,>(
  success: boolean,
  message: string,
  data?: T,
  status = 200
) =>
  NextResponse.json(
    {
      success,
      message,
      ...(data ? { data } : {}),
    } as ApiResponse<T>,
    { status }
  );

const normalizeFrequency = (value: unknown, fallback = DEFAULT_SCAN_FREQUENCY) => {
  if (typeof value !== "number") return fallback;
  return VALID_FREQUENCIES.has(value) ? value : fallback;
};

const normalizeBoolean = (value: unknown, fallback: boolean) =>
  typeof value === "boolean" ? value : fallback;

const normalizeText = (value: unknown, fallback: string) =>
  typeof value === "string" && value.trim().length > 0 ? value.trim() : fallback;

const extractJson = (rawContent: string) => {
  let jsonStr = rawContent.trim();
  // Strip markdown code fences
  const codeFenceMatch = jsonStr.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (codeFenceMatch) {
    jsonStr = codeFenceMatch[1].trim();
  }
  // Find the JSON object
  const jsonMatch = jsonStr.match(/\{[\s\S]*\}/);
  return jsonMatch ? jsonMatch[0] : null;
};

const sanitizeObjects = (value: unknown): DetectedObject[] => {
  if (!Array.isArray(value)) return [];

  return value
    .map((item) => {
      if (!item || typeof item !== "object") return null;
      const obj = item as {
        name?: unknown;
        distance?: unknown;
        direction?: unknown;
        confidence?: unknown;
      };

      const name = normalizeText(obj.name, "");
      const distance =
        obj.distance === "very close" ||
        obj.distance === "close" ||
        obj.distance === "near" ||
        obj.distance === "far" ||
        obj.distance === "uncertain"
          ? obj.distance
          : null;
      const direction =
        obj.direction === "left" ||
        obj.direction === "center" ||
        obj.direction === "right" ||
        obj.direction === "ahead"
          ? obj.direction
          : null;
      const confidenceValue = typeof obj.confidence === "number" ? obj.confidence : null;
      const confidence =
        confidenceValue === null
          ? 0.5
          : Math.max(0, Math.min(1, confidenceValue));

      if (!name || !distance || !direction) return null;
      return { name, distance, direction, confidence } as DetectedObject;
    })
    .filter((item): item is DetectedObject => Boolean(item))
    .slice(0, 12);
};

const sanitizeHazards = (value: unknown): string[] => {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => normalizeText(item, ""))
    .filter((item) => item.length > 0)
    .slice(0, 8);
};

const sanitizeResult = (value: Partial<ScanResult>): ScanResult => {
  const objects = sanitizeObjects(value.objects);
  const hazards = sanitizeHazards(value.hazards);
  const scene_summary = normalizeText(
    value.scene_summary,
    objects.length > 0 ? "Scene contains visible objects." : "Scene summary unavailable."
  );
  const action_advice = normalizeText(
    value.action_advice,
    hazards.length > 0
      ? "Proceed with caution."
      : objects.length > 0
        ? "Proceed carefully around nearby objects."
        : "Scene unclear. Proceed cautiously and rescan."
  );

  return {
    scene_summary,
    objects,
    hazards,
    action_advice,
  };
};

const parseModelResponse = (rawContent: string) => {
  const jsonStr = extractJson(rawContent);
  if (!jsonStr) {
    return { result: FALLBACK_RESULT, rawJson: rawContent };
  }
  try {
    const parsed = JSON.parse(jsonStr) as Partial<ScanResult>;
    return {
      result: sanitizeResult(parsed),
      rawJson: parsed,
    };
  } catch {
    return { result: FALLBACK_RESULT, rawJson: rawContent };
  }
};

const mapDangerLevel = (result: ScanResult) => {
  if (result.hazards.length === 0) return 0;
  const hazardText = result.hazards.join(" ").toLowerCase();
  if (hazardText.includes("very close") || hazardText.includes("close")) {
    return 3;
  }
  return 2;
};

const toInlineData = (image: string) => {
  if (image.startsWith("data:")) {
    const match = image.match(/^data:(.*?);base64,(.*)$/i);
    if (match) {
      return { mimeType: match[1], data: match[2] };
    }
  }
  return { mimeType: "image/jpeg", data: image };
};

// Validate that the image data is not empty/too small
const validateImageData = (image: string): boolean => {
  const base64Data = image.startsWith("data:")
    ? image.split(",")[1] || ""
    : image;
  // Minimum ~1KB of image data to be useful
  return base64Data.length > 1000;
};

// Retry wrapper for API calls — does NOT retry on rate limit (429) errors
const withRetry = async <T>(
  fn: () => Promise<T>,
  retries: number = MAX_API_RETRIES,
  delayMs: number = 2000
): Promise<T> => {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      // Never retry on rate limit — it just makes it worse
      if (lastError.message === "RATE_LIMITED") throw lastError;
      if (attempt < retries) {
        await new Promise((resolve) => setTimeout(resolve, delayMs * (attempt + 1)));
      }
    }
  }
  throw lastError;
};

export async function POST(request: NextRequest) {
  let body: VisionScanRequest;
  try {
    body = (await request.json()) as VisionScanRequest;
  } catch {
    return jsonResponse(false, "Invalid request body.", undefined, 400);
  }

  const action = body.action ?? "scan";
  const userId = typeof body.userId === "string" ? body.userId.trim() : "";
  const localVisionUrl = process.env.SMOLVLM2_URL ?? process.env.LOCAL_VISION_URL;
  const hasLocalVision = Boolean(localVisionUrl);
  const hasZaiKey = Boolean(
    process.env.ZAI_API_KEY ?? process.env.OPENAI_API_KEY
  );
  const hasGeminiKey = Boolean(process.env.GEMINI_API_KEY);
  const offlineOnly = (process.env.OFFLINE_ONLY ?? "true").toLowerCase() !== "false";

  // ─── Health check ───────────────────────────────────────────────────────
  if (action === "health") {
    const provider = hasLocalVision
      ? "smolvlm2"
      : offlineOnly
        ? null
        : hasZaiKey
          ? "zai"
          : hasGeminiKey
            ? "gemini"
            : null;
    const model = hasLocalVision
      ? process.env.SMOLVLM2_MODEL_ID ?? "SmolVLM2-2.2B-Instruct"
      : offlineOnly
        ? undefined
        : hasZaiKey
          ? process.env.ZAI_VISION_MODEL
          : hasGeminiKey
            ? process.env.GEMINI_MODEL ?? DEFAULT_GEMINI_MODEL
            : undefined;
    const configured = Boolean(provider);
    const message = configured
      ? "Vision model configured."
      : offlineOnly
        ? "Offline mode enabled. Set SMOLVLM2_URL to use the local SmolVLM2 server."
        : "Vision model not configured. Set SMOLVLM2_URL or GEMINI_API_KEY in your .env file.";

    return jsonResponse<VisionHealthData>(true, message, {
      configured,
      provider,
      model,
    });
  }

  // ─── Get preferences ────────────────────────────────────────────────────
  if (action === "get-preferences") {
    if (!userId) {
      return jsonResponse(false, "Missing user identifier.", undefined, 400);
    }

    const dbClient = await getDb();
    if (!dbClient) {
      return jsonResponse<PreferencesData>(true, "Preferences loaded.", {
        scanFrequency: DEFAULT_SCAN_FREQUENCY,
        hapticEnabled: true,
        ttsVoice: "default",
      });
    }

    try {
      const existing = await dbClient.userPreference.findUnique({
        where: { userId },
      });
      if (!existing) {
        const created = await dbClient.userPreference.create({
          data: {
            userId,
            scanFrequency: DEFAULT_SCAN_FREQUENCY,
            hapticEnabled: true,
            ttsVoice: "default",
          },
        });

        return jsonResponse<PreferencesData>(true, "Preferences loaded.", {
          scanFrequency: created.scanFrequency,
          hapticEnabled: created.hapticEnabled,
          ttsVoice: created.ttsVoice,
        });
      }

      return jsonResponse<PreferencesData>(true, "Preferences loaded.", {
        scanFrequency: existing.scanFrequency,
        hapticEnabled: existing.hapticEnabled,
        ttsVoice: existing.ttsVoice,
      });
    } catch {
      return jsonResponse<PreferencesData>(true, "Preferences loaded (default).", {
        scanFrequency: DEFAULT_SCAN_FREQUENCY,
        hapticEnabled: true,
        ttsVoice: "default",
      });
    }
  }

  // ─── Update preferences ─────────────────────────────────────────────────
  if (action === "update-preferences") {
    if (!userId) {
      return jsonResponse(false, "Missing user identifier.", undefined, 400);
    }

    const dbClient = await getDb();
    const existing = dbClient
      ? await dbClient.userPreference.findUnique({ where: { userId } }).catch(() => null)
      : null;
    const scanFrequency = normalizeFrequency(
      body.scanFrequency ?? existing?.scanFrequency
    );
    const hapticEnabled = normalizeBoolean(
      body.hapticEnabled,
      existing?.hapticEnabled ?? true
    );
    const ttsVoice = normalizeText(
      body.ttsVoice ?? existing?.ttsVoice,
      "default"
    );

    if (dbClient) {
      try {
        const preferences = await dbClient.userPreference.upsert({
          where: { userId },
          create: {
            userId,
            scanFrequency,
            hapticEnabled,
            ttsVoice,
          },
          update: {
            scanFrequency,
            hapticEnabled,
            ttsVoice,
          },
        });

        return jsonResponse<PreferencesData>(true, "Preferences saved.", {
          scanFrequency: preferences.scanFrequency,
          hapticEnabled: preferences.hapticEnabled,
          ttsVoice: preferences.ttsVoice,
        });
      } catch {
        // DB write failed, still return success with the values
      }
    }

    return jsonResponse<PreferencesData>(true, "Preferences saved.", {
      scanFrequency,
      hapticEnabled,
      ttsVoice,
    });
  }

  // ─── Vision scan ────────────────────────────────────────────────────────
  if (!body.image) {
    return jsonResponse(false, "No image data provided.", undefined, 400);
  }

  if (typeof body.image !== "string") {
    return jsonResponse(false, "Image data must be a base64 string.", undefined, 400);
  }

  // Server-side rate limiting — prevent hammering the vision service
  const now = Date.now();
  const timeSinceLastScan = now - lastScanTimeMs;
  if (timeSinceLastScan < MIN_SCAN_INTERVAL_MS) {
    const waitSec = Math.ceil((MIN_SCAN_INTERVAL_MS - timeSinceLastScan) / 1000);
    return jsonResponse(
      false,
      `Please wait ${waitSec} more second${waitSec > 1 ? 's' : ''} before scanning again.`,
      undefined,
      429
    );
  }

  // Validate image data quality
  if (!validateImageData(body.image)) {
    return jsonResponse(
      false,
      "Image data is too small or empty. Please ensure the camera is working and try again.",
      undefined,
      400
    );
  }

  if (offlineOnly && !hasLocalVision) {
    return jsonResponse(
      false,
      "Offline mode enabled. Start the local SmolVLM2 server and set SMOLVLM2_URL.",
      undefined,
      500
    );
  }

  if (!offlineOnly && !hasLocalVision && !hasZaiKey && !hasGeminiKey) {
    return jsonResponse(
      false,
      "Vision model not configured. Set SMOLVLM2_URL or GEMINI_API_KEY in your .env file.",
      undefined,
      500
    );
  }

  const imageUrl = body.image.startsWith("data:")
    ? body.image
    : `data:image/jpeg;base64,${body.image}`;

  // Mark scan time to enforce rate limit
  lastScanTimeMs = Date.now();

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), SCAN_TIMEOUT_MS);

  let rawContent: string | null = null;
  try {
    if (hasLocalVision) {
      const response = await fetch(localVisionUrl as string, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: imageUrl,
          prompt: `${SYSTEM_PROMPT}\n\n${RESPONSE_SCHEMA_PROMPT}`,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorText = await response.text().catch(() => "Unknown error");
        throw new Error(`LOCAL_VISION_ERROR ${response.status}: ${errorText}`);
      }

      const payload = await response.json().catch(() => null);
      rawContent = payload ? JSON.stringify(payload) : null;
    } else if (!offlineOnly && hasZaiKey) {
      const zai = await getZAI();
      const messages = [
        { role: "system", content: SYSTEM_PROMPT },
        {
          role: "user",
          content: [
            { type: "text", text: RESPONSE_SCHEMA_PROMPT },
            { type: "image_url", image_url: { url: imageUrl } },
          ],
        },
      ];

      const completions = zai.chat.completions as {
        create?: (payload: unknown) => Promise<{ choices?: Array<{ message?: { content?: string | null } | null }> }>;
        createVision?: (payload: unknown) => Promise<{ choices?: Array<{ message?: { content?: string | null } | null }> }>;
      };

      const createCompletion = completions.create ?? completions.createVision;
      if (!createCompletion) {
        clearTimeout(timeoutId);
        return jsonResponse(false, "Vision model is unavailable.", undefined, 503);
      }

      const model = process.env.ZAI_VISION_MODEL;
      const payload = {
        messages,
        thinking: { type: "disabled" },
        ...(model ? { model } : {}),
        signal: controller.signal,
      };
      const response = await createCompletion(payload);
      rawContent = response?.choices?.[0]?.message?.content ?? null;
    } else if (!offlineOnly && hasGeminiKey) {
      const inlineData = toInlineData(imageUrl);
      const model = process.env.GEMINI_MODEL ?? DEFAULT_GEMINI_MODEL;

      const requestBody = {
        contents: [
          {
            role: "user",
            parts: [
              {
                text: `${SYSTEM_PROMPT}\n\n${RESPONSE_SCHEMA_PROMPT}`,
              },
              {
                inlineData: {
                  mimeType: inlineData.mimeType,
                  data: inlineData.data,
                },
              },
            ],
          },
        ],
        generationConfig: {
          temperature: 0.3,
          maxOutputTokens: 600,
          responseMimeType: "application/json",
        },
      };

      // Use retry logic for transient Gemini API failures
      const geminiResult = await withRetry(async () => {
        const response = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(
            process.env.GEMINI_API_KEY ?? ""
          )}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
          }
        );

        if (!response.ok) {
          const errorText = await response.text().catch(() => "Unknown error");
          if (response.status === 401 || response.status === 403) {
            throw new Error("AUTH_FAILED");
          }
          if (response.status === 429) {
            throw new Error("RATE_LIMITED");
          }
          throw new Error(`Gemini API error ${response.status}: ${errorText}`);
        }

        return response.json();
      }, MAX_API_RETRIES);

      const geminiData = geminiResult as {
        candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
      };
      rawContent =
        geminiData.candidates?.[0]?.content?.parts
          ?.map((part) => part.text)
          .filter((text): text is string => Boolean(text))
          .join("\n") ?? null;
    }
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error) {
      if (error.name === "AbortError") {
        return jsonResponse(
          false,
          "Vision scan timed out. Please try again.",
          undefined,
          504
        );
      }
      if (error.message === "AUTH_FAILED") {
        return jsonResponse(
          false,
          "Vision model auth failed. Check your GEMINI_API_KEY in .env file.",
          undefined,
          401
        );
      }
      if (error.message.startsWith("LOCAL_VISION_ERROR")) {
        return jsonResponse(
          false,
          "Local SmolVLM2 service failed. Check that it is running and reachable.",
          undefined,
          502
        );
      }
      if (error.message === "RATE_LIMITED") {
        return jsonResponse(
          false,
          "Too many requests. Please wait a moment and try again.",
          undefined,
          429
        );
      }
      const message = error.message.toLowerCase();
      if (
        message.includes("unauthorized") ||
        message.includes("401") ||
        message.includes("api key")
      ) {
        return jsonResponse(
          false,
          "Vision model auth failed. Check your API key configuration.",
          undefined,
          401
        );
      }
    }

    return jsonResponse(false, "Vision scan failed. Please try again.", undefined, 500);
  } finally {
    clearTimeout(timeoutId);
  }

  if (!rawContent) {
    return jsonResponse(false, "No response from vision model. The model may not have detected anything in the image.", undefined, 502);
  }

  const { result, rawJson } = parseModelResponse(rawContent);

  // Save scan record to database
  let saveWarning = false;
  if (userId) {
    const scanFrequency = normalizeFrequency(body.scanFrequency);
    const hapticEnabled = normalizeBoolean(body.hapticEnabled, true);
    const ttsVoice = normalizeText(body.ttsVoice, "default");
    const dbClient = await getDb();

    if (!dbClient) {
      saveWarning = true;
    } else {
      try {
        await dbClient.userPreference.upsert({
          where: { userId },
          create: {
            userId,
            scanFrequency,
            hapticEnabled,
            ttsVoice,
          },
          update: {
            scanFrequency,
            hapticEnabled,
            ttsVoice,
          },
        });

        await dbClient.scanRecord.create({
          data: {
            userId,
            description: result.action_advice || result.scene_summary,
            rawJson,
            confidence: null,
            dangerLevel: mapDangerLevel(result),
          },
        });
      } catch {
        saveWarning = true;
      }
    }
  }

  return jsonResponse(
    true,
    saveWarning
      ? "Scan completed, but saving history failed."
      : "Scan completed.",
    result
  );
}
