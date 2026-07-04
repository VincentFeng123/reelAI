import { JobSnapshot, Settings } from "./types";

// All relative URLs → same-origin in prod, proxied in dev, LAN-friendly on a phone.
export async function createJob(
  url: string,
  topic: string,
  settings: Partial<Settings>,
): Promise<string> {
  const r = await fetch("/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, topic, settings }),
  });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    const msg =
      (Array.isArray(e?.detail) && e.detail[0]?.msg) ||
      (typeof e?.detail === "string" && e.detail) ||
      "Could not start the job.";
    throw new Error(msg.replace(/^Value error,\s*/i, ""));
  }
  const d = await r.json();
  return d.job_id as string;
}

export async function getJob(id: string): Promise<JobSnapshot> {
  const r = await fetch(`/jobs/${id}`);
  if (!r.ok) throw new Error("Job not found");
  return r.json();
}

export async function exportClip(jobId: string, n: number): Promise<{ path: string }> {
  const r = await fetch(`/jobs/${jobId}/clips/${n}/export`, { method: "POST" });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error((typeof e?.detail === "string" && e.detail) || "Export failed");
  }
  return r.json();
}

export const clipDownloadUrl = (path: string) => `${path}?download=1`;
export const zipUrl = (jobId: string) => `/jobs/${jobId}/zip`;
