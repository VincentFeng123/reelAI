import { useState } from "react";
import { Clip } from "../types";
import { exportClip, clipDownloadUrl } from "../api";

function fmt(t: number): string {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// Only a curated, user-facing subset of internal warning strings is surfaced; the rest
// (merged_overlap, single_idea_unverified, missing_context_card, unjudged, …) stay hidden.
const WARNING_LABELS: Record<string, string> = {
  unverified_judge_concerns: "unverified",
  trimmed_start: "trimmed",
  capped_max_duration: "length-capped",
};

function userWarnings(warnings?: string[]): string[] {
  const seen = new Set<string>();
  for (const w of warnings || []) {
    const label = WARNING_LABELS[w];
    if (label) seen.add(label);
  }
  return [...seen];
}

export function ClipCard({ clip, jobId }: { clip: Clip; jobId: string }) {
  const [path, setPath] = useState<string | null>(clip.path);
  const [exporting, setExporting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const onExport = async () => {
    setExporting(true);
    setErr(null);
    try {
      const r = await exportClip(jobId, clip.n);
      setPath(r.path);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Export failed");
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="glass overflow-hidden flex flex-col animate-floatIn">
      <div className="aspect-video bg-black/40">
        <iframe
          className="w-full h-full"
          src={clip.embed_url}
          title={`Clip ${clip.n}`}
          loading="lazy"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        />
      </div>
      <div className="p-4 flex flex-col gap-2.5 flex-1">
        {clip.title && (
          <h3 className="text-sm font-medium text-white/90 leading-snug">{clip.title}</h3>
        )}
        <div className="flex items-center justify-between gap-2">
          <span className="tag">{clip.facet.replace(/_/g, " ")}</span>
          <span className="text-xs text-white/40 tabular-nums">
            {fmt(clip.start)}–{fmt(clip.end)} · {clip.duration.toFixed(0)}s
          </span>
        </div>
        {(clip.final_quality != null ||
          clip.ship_flagged ||
          (clip.prerequisite_clips && clip.prerequisite_clips.length > 0) ||
          userWarnings(clip.warnings).length > 0) && (
          <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
            {clip.final_quality != null && (
              <span
                className={
                  clip.final_quality >= 0.7 ? "text-emerald-300/80" : "text-white/45"
                }
              >
                {clip.final_quality >= 0.7 ? "★ High" : "OK"}
              </span>
            )}
            {clip.prerequisite_clips?.map((p) => (
              <span key={`pre-${p}`} className="tag !text-sky-200/80 !bg-sky-400/10">
                ▶ watch clip {p} first
              </span>
            ))}
            {(clip.ship_flagged || userWarnings(clip.warnings).length > 0) &&
              (userWarnings(clip.warnings).length > 0
                ? userWarnings(clip.warnings)
                : ["unverified"]
              ).map((w) => (
                <span key={`warn-${w}`} className="tag !text-amber-200/80 !bg-amber-400/10">
                  ⚠ {w}
                </span>
              ))}
          </div>
        )}
        {clip.context_card && (
          <p className="text-xs text-white/55 leading-snug border-l-2 border-white/15 pl-2 italic">
            {clip.context_card}
          </p>
        )}
        <p className="text-sm text-white/70 leading-snug flex-1">{clip.reason}</p>
        {clip.notes && clip.notes.length > 0 && (
          <ul className="text-[11px] text-white/40 leading-snug list-none flex flex-col gap-0.5">
            {clip.notes.map((note, i) => (
              <li key={`note-${i}`}>· {note}</li>
            ))}
          </ul>
        )}
        {path ? (
          <a href={clipDownloadUrl(path)} download className="btn-ghost text-center text-sm">
            ↓ Download .mp4
          </a>
        ) : (
          <button onClick={onExport} disabled={exporting} className="btn-ghost text-sm">
            {exporting ? "Exporting… (downloading section)" : "Export .mp4"}
          </button>
        )}
        {err && <p className="text-xs text-rose-300">{err}</p>}
      </div>
    </div>
  );
}
