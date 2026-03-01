"use client";

import { type DragEvent, type FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { uploadMaterial } from "@/lib/api";

const MATERIAL_SEEDS_STORAGE_KEY = "studyreels-material-seeds";
const GENERATION_MODE_STORAGE_KEY = "studyreels-generation-mode";
const MAX_MATERIAL_SEEDS = 120;
const MAX_SEED_TEXT_CHARS = 16000;
type GenerationMode = "slow" | "fast";

type MaterialSeed = {
  topic?: string;
  text?: string;
  title: string;
  updatedAt: number;
};

function parseMaterialSeeds(raw: string | null): Record<string, MaterialSeed> {
  if (!raw) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    const normalized: Record<string, MaterialSeed> = {};
    for (const [id, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (!id || typeof id !== "string") {
        continue;
      }
      if (!value || typeof value !== "object" || Array.isArray(value)) {
        continue;
      }
      const seed = value as Record<string, unknown>;
      const title = String(seed.title || "").trim();
      if (!title) {
        continue;
      }
      normalized[id] = {
        topic: String(seed.topic || "").trim() || undefined,
        text: String(seed.text || "").trim() || undefined,
        title,
        updatedAt: Number(seed.updatedAt) || 0,
      };
    }
    return normalized;
  } catch {
    return {};
  }
}

type UploadPanelProps = {
  onMaterialCreated?: (params: {
    materialId: string;
    title: string;
    topic?: string;
    generationMode: GenerationMode;
  }) => void | Promise<void>;
};

function buildMaterialTitle(params: { topic: string; text: string; fileName: string }): string {
  const topic = params.topic.trim();
  if (topic) {
    return topic;
  }
  if (params.fileName.trim()) {
    return params.fileName.trim();
  }
  const snippet = params.text.trim().replace(/\s+/g, " ").slice(0, 58);
  if (snippet) {
    return snippet;
  }
  return "New Study Session";
}

export function UploadPanel({ onMaterialCreated }: UploadPanelProps) {
  const router = useRouter();
  const [topic, setTopic] = useState("");
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | undefined>();
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generationMode, setGenerationMode] = useState<GenerationMode>("slow");
  const selectedFileName = file?.name ?? "";

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const saved = window.localStorage.getItem(GENERATION_MODE_STORAGE_KEY);
    if (saved === "fast" || saved === "slow") {
      setGenerationMode(saved);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(GENERATION_MODE_STORAGE_KEY, generationMode);
  }, [generationMode]);

  const disabled = useMemo(() => {
    return loading || (!file && !topic.trim() && !text.trim());
  }, [file, loading, text, topic]);

  const onSubmit = useCallback(async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const topicValue = topic.trim();
      const textValue = text.trim();
      const title = buildMaterialTitle({
        topic: topicValue,
        text: textValue,
        fileName: file?.name ?? "",
      });
      const material = await uploadMaterial({
        text: textValue || undefined,
        file,
        subjectTag: topicValue || undefined,
      });
      if (typeof window !== "undefined") {
        const seeds = parseMaterialSeeds(window.localStorage.getItem(MATERIAL_SEEDS_STORAGE_KEY));
        seeds[material.material_id] = {
          topic: topicValue || undefined,
          text: textValue ? textValue.slice(0, MAX_SEED_TEXT_CHARS) : undefined,
          title,
          updatedAt: Date.now(),
        };
        const ordered = Object.entries(seeds)
          .sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0))
          .slice(0, MAX_MATERIAL_SEEDS);
        window.localStorage.setItem(MATERIAL_SEEDS_STORAGE_KEY, JSON.stringify(Object.fromEntries(ordered)));
      }
      if (onMaterialCreated) {
        await onMaterialCreated({
          materialId: material.material_id,
          title,
          topic: topicValue || undefined,
          generationMode,
        });
      }
      router.push(`/feed?material_id=${material.material_id}&generation_mode=${generationMode}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something failed");
    } finally {
      setLoading(false);
    }
  }, [file, generationMode, onMaterialCreated, router, text, topic]);

  const onFileDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDraggingFile(false);
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) {
      setFile(dropped);
    }
  };

  return (
    <form
      onSubmit={onSubmit}
      className="flex h-full w-full flex-col overflow-x-visible overflow-y-auto px-6 py-6 pb-24 md:overflow-hidden md:px-10 md:py-8 md:pb-8 lg:px-5"
    >
      <header className="mb-5 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="text-center text-xs font-semibold uppercase tracking-[0.18em] text-white/55 md:text-left">Study Feed</p>
          <div className="mt-8 md:mt-2">
            <h1 className="text-3xl font-bold tracking-tight md:text-5xl">StudyReels</h1>
            <p className="mt-2 max-w-2xl text-sm text-white/68">
              Type a topic, paste text, upload a file, or combine all of them. The feed starts with short reels and keeps expanding as you scroll.
            </p>
          </div>
        </div>
      </header>

      <div className="mt-10 grid gap-4 md:mt-0 md:grid-cols-2">
        <div>
          <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.12em] text-white/70">Topic</label>
          <input
            className="h-12 w-full rounded-2xl border border-white/30 bg-black/45 px-4 text-sm text-white outline-none transition placeholder:text-white/40 focus:border-white/65"
            placeholder="e.g. linear regression"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
          />
        </div>

        <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_190px] sm:items-end">
          <div className="min-w-0 flex-1">
            <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.12em] text-white/70">File</label>
            <input
              id="material-file"
              className="sr-only"
              type="file"
              accept=".pdf,.docx,.txt"
              onChange={(e) => setFile(e.target.files?.[0])}
            />
            <label
              htmlFor="material-file"
              onDragOver={(event) => {
                event.preventDefault();
                setIsDraggingFile(true);
              }}
              onDragLeave={() => setIsDraggingFile(false)}
              onDrop={onFileDrop}
              className={`flex h-12 w-full cursor-pointer items-center rounded-2xl border bg-black/45 px-4 text-sm text-white outline-none transition ${
                isDraggingFile ? "border-white/70" : "border-white/30"
              }`}
            >
              <span className={`truncate ${selectedFileName ? "text-white" : "text-white/40"}`}>
                {selectedFileName || "Drag & drop or choose file"}
              </span>
            </label>
          </div>

          <div className="w-full">
            <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.12em] text-white/70">Generation Speed</label>
            <div className="relative grid h-12 grid-cols-2 items-center rounded-2xl border border-white/25 bg-black/45 p-1">
              <span
                aria-hidden="true"
                className={`pointer-events-none absolute bottom-1 left-1 top-1 w-[calc(50%-4px)] rounded-xl bg-white transition-transform duration-300 ease-out ${
                  generationMode === "fast" ? "translate-x-full" : "translate-x-0"
                }`}
              />
              <button
                type="button"
                onClick={() => setGenerationMode("slow")}
                className={`relative z-10 rounded-xl px-2 py-2 text-[10px] font-semibold uppercase tracking-[0.06em] transition-colors duration-200 ${
                  generationMode === "slow" ? "text-black" : "text-white/80"
                }`}
                aria-pressed={generationMode === "slow"}
              >
                Slow
              </button>
              <button
                type="button"
                onClick={() => setGenerationMode("fast")}
                className={`relative z-10 rounded-xl px-2 py-2 text-[10px] font-semibold uppercase tracking-[0.06em] transition-colors duration-200 ${
                  generationMode === "fast" ? "text-black" : "text-white/80"
                }`}
                aria-pressed={generationMode === "fast"}
              >
                Fast
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 flex-[1.15] min-h-[220px]">
        <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.12em] text-white/70">Source Text</label>
        <textarea
          className="h-full min-h-[220px] w-full resize-none overflow-y-auto rounded-2xl border border-white/30 bg-black/45 p-5 text-sm leading-relaxed text-white outline-none transition placeholder:text-white/40 focus:border-white/65"
          placeholder="Paste notes, textbook text, or any material here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
      </div>

      <div className="mt-2 shrink-0 flex flex-col gap-2 md:mt-6 md:gap-3 md:flex-row md:items-center md:justify-between">
        <p className="min-h-5 text-sm text-white/80">{error ?? ""}</p>
        <button
          type="submit"
          disabled={disabled}
          className="mt-0 w-full rounded-2xl border border-white/30 bg-white px-7 py-3 text-sm font-bold text-black transition hover:bg-white/92 disabled:cursor-not-allowed disabled:opacity-60 md:mt-4 md:w-auto"
        >
          {loading ? "Starting..." : "Start Learning"}
        </button>
      </div>
    </form>
  );
}
