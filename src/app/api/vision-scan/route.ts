import { NextRequest, NextResponse } from "next/server";
import ZAI from "z-ai-web-dev-sdk";

interface Threat {
  label: string;
  urgency: "high" | "medium" | "low";
  position: "ahead" | "left" | "right";
}

interface ScanResult {
  threats: Threat[];
  voiceAlert: string;
  safeAction: string;
  summary: string;
  threatLevel: "clear" | "caution" | "danger";
}

const SYSTEM_PROMPT = `You are a blind person's navigation assistant. You analyze camera frames from a person walking outdoors or indoors and identify potential hazards.

Analyze the image and respond with ONLY a valid JSON object (no markdown, no code fences, no extra text). The JSON must have this exact structure:

{
  "threats": [
    { "label": "description of object", "urgency": "high|medium|low", "position": "ahead|left|right" }
  ],
  "voiceAlert": "one short spoken sentence under 15 words describing the most urgent threat or the scene",
  "safeAction": "5 words max describing what to do",
  "summary": "one sentence describing the overall scene",
  "threatLevel": "clear|caution|danger"
}

Rules:
- If the path ahead is clear and safe, return an empty threats array and threatLevel "clear"
- urgency "high" = immediate danger (vehicle, drop, fire, obstacle very close)
- urgency "medium" = potential concern (pole, low branch, uneven surface, person nearby)
- urgency "low" = minor awareness (pothole far away, sign, parked car)
- position should be relative to the camera wearer's perspective
- voiceAlert MUST be under 15 words and speakable — this will be read aloud
- safeAction MUST be 5 words max
- summary MUST be exactly 1 sentence
- If there are no threats, voiceAlert should say something reassuring about the path being clear
- Respond ONLY with the JSON, nothing else`;

let zaiInstance: Awaited<ReturnType<typeof ZAI.create>> | null = null;

async function getZAI() {
  if (!zaiInstance) {
    zaiInstance = await ZAI.create();
  }
  return zaiInstance;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { image } = body;

    if (!image) {
      return NextResponse.json(
        { error: "No image data provided" },
        { status: 400 }
      );
    }

    const zai = await getZAI();

    const response = await zai.chat.completions.createVision({
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: SYSTEM_PROMPT,
            },
            {
              type: "image_url",
              image_url: {
                url: image.startsWith("data:")
                  ? image
                  : `data:image/jpeg;base64,${image}`,
              },
            },
          ],
        },
      ],
      thinking: { type: "disabled" },
    });

    const rawContent = response.choices[0]?.message?.content;

    if (!rawContent) {
      return NextResponse.json(
        { error: "No response from vision model" },
        { status: 500 }
      );
    }

    // Try to parse the JSON from the response - it may be wrapped in markdown code fences
    let jsonStr = rawContent.trim();

    // Remove markdown code fences if present
    const codeFenceMatch = jsonStr.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    if (codeFenceMatch) {
      jsonStr = codeFenceMatch[1].trim();
    }

    // Try to find JSON object in the response
    const jsonMatch = jsonStr.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      jsonStr = jsonMatch[0];
    }

    let result: ScanResult;
    try {
      result = JSON.parse(jsonStr);
    } catch {
      // If parsing fails, create a safe fallback
      result = {
        threats: [],
        voiceAlert: "Unable to analyze scene. Please scan again.",
        safeAction: "Scan again",
        summary: "Vision analysis returned an unexpected response.",
        threatLevel: "caution",
      };
    }

    // Validate and sanitize the result
    if (!Array.isArray(result.threats)) result.threats = [];
    result.threats = result.threats.slice(0, 10); // Limit threats

    if (!result.voiceAlert || typeof result.voiceAlert !== "string") {
      result.voiceAlert = "Scene analysis complete.";
    }
    if (!result.safeAction || typeof result.safeAction !== "string") {
      result.safeAction = "Proceed carefully";
    }
    if (!result.summary || typeof result.summary !== "string") {
      result.summary = "Unable to generate summary.";
    }
    if (!["clear", "caution", "danger"].includes(result.threatLevel)) {
      result.threatLevel = "caution";
    }

    return NextResponse.json({ success: true, data: result });
  } catch (error) {
    console.error("Vision scan error:", error);
    return NextResponse.json(
      {
        error: "Scan failed — retrying",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
