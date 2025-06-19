"use server";

import path from "path";
import { writeFile } from "fs/promises";

type AnalysisResult = { naked: boolean; kids: boolean };

export async function analyzeImage(
  formData: FormData
): Promise<{ result?: AnalysisResult; error?: string }> {
  try {
    const file = formData.get("image");
    if (!file || typeof file === "string") {
      return { error: "Invalid or missing image file." };
    }

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    // Optional: Save the file temporarily to /tmp (for debugging or logging)
    const tempPath = path.join("/tmp", file.name);
    await writeFile(tempPath, buffer);

    // Send request to Ollama
    const res = await fetch(`${process.env.OLLAMA_PATH_API}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gemma3:12b",
        prompt:
          'Analyze this image and respond only with plain JSON, no extra text: {"naked": true|false, "kids": true|false}',
        role: "user",
        stream: false,
        temperature: 0.2,
        top_p: 0.9,
        images: [buffer.toString("base64")],
      }),
    });

    if (!res.ok) {
      console.error("HTTP error from Ollama:", res.status, res.statusText);
      return { error: `Server error: ${res.status}` };
    }

    const data = await res.json();
    const raw = (data.response || "").trim();

    // Clean markdown code block if present
    const jsonClean = raw
      .replace(/```json/, "")
      .replace(/```/, "")
      .replace(/[\r\n]/g, "")
      .trim();

    // Basic validation before attempting to parse
    if (!jsonClean || jsonClean.length < 10 || !jsonClean.includes("{")) {
      console.error("Empty or malformed model response:", raw);
      return { error: "Model returned an empty or invalid response." };
    }

    const parsed = JSON.parse(jsonClean);

    // Check if the parsed result matches expected structure
    if (typeof parsed.naked !== "boolean" || typeof parsed.kids !== "boolean") {
      throw new Error("JSON structure is not valid.");
    }

    return { result: parsed as AnalysisResult };
  } catch (error) {
    console.error("Error in analyzeImage:", error);
    return { error: "An error occurred while processing the image." };
  }
}
