import type { RunCreateResponse } from "./types";

export async function startRun(input: string): Promise<RunCreateResponse> {
  const response = await fetch("/api/runs", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ input }),
  });

  if (!response.ok) {
    const message = await readError(response);
    throw new Error(message);
  }

  return response.json() as Promise<RunCreateResponse>;
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    return payload.detail || `Request failed with ${response.status}`;
  } catch {
    return `Request failed with ${response.status}`;
  }
}
