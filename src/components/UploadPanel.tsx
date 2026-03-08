"use client";

import { type DragEvent, type FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { uploadMaterial } from "@/lib/api";
import { type GenerationMode, type SearchInputMode, readStudyReelsSettings, subscribeToStudyReelsSettings } from "@/lib/settings";

const MATERIAL_SEEDS_STORAGE_KEY = "studyreels-material-seeds";
const MATERIAL_GROUPS_STORAGE_KEY = "studyreels-material-groups";
const MAX_MATERIAL_SEEDS = 120;
const MAX_MATERIAL_GROUPS = 80;
const MAX_SEED_TEXT_CHARS = 16000;
type InputMode = SearchInputMode;

const INPUT_MODE_OPTIONS: Array<{ value: InputMode; label: string }> = [
  { value: "topic", label: "Topic" },
  { value: "source", label: "Text" },
  { value: "file", label: "File Upload" },
];

type MaterialSeed = {
  topic?: string;
  text?: string;
  title: string;
  updatedAt: number;
};

type MaterialGroup = {
  materialIds: string[];
  title?: string;
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

function parseMaterialGroups(raw: string | null): Record<string, MaterialGroup> {
  if (!raw) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    const normalized: Record<string, MaterialGroup> = {};
    for (const [id, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (!id || typeof id !== "string") {
        continue;
      }
      if (!value || typeof value !== "object" || Array.isArray(value)) {
        continue;
      }
      const group = value as Record<string, unknown>;
      const materialIds = Array.isArray(group.materialIds)
        ? Array.from(
            new Set(
              group.materialIds
                .map((row) => String(row || "").trim())
                .filter(Boolean),
            ),
          )
        : [];
      if (materialIds.length === 0) {
        continue;
      }
      normalized[id] = {
        materialIds,
        title: typeof group.title === "string" && group.title.trim() ? group.title.trim() : undefined,
        updatedAt: Number(group.updatedAt) || 0,
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
  onScrollOffsetChange?: (isOffset: boolean) => void;
  onScrollGesture?: () => void;
  onScrollabilityChange?: (isScrollable: boolean) => void;
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

export function UploadPanel({ onMaterialCreated, onScrollOffsetChange, onScrollGesture, onScrollabilityChange }: UploadPanelProps) {
  const router = useRouter();
  const touchStartYRef = useRef<number | null>(null);
  const formRef = useRef<HTMLFormElement | null>(null);
  const [topics, setTopics] = useState<string[]>([""]);
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | undefined>();
  const [inputMode, setInputMode] = useState<InputMode>("source");
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generationMode, setGenerationMode] = useState<GenerationMode>("slow");
  const selectedFileName = file?.name ?? "";

  useEffect(() => {
    const saved = readStudyReelsSettings();
    setGenerationMode(saved.generationMode);
    setInputMode(saved.defaultInputMode);
    return subscribeToStudyReelsSettings((next) => {
      setGenerationMode(next.generationMode);
      setInputMode(next.defaultInputMode);
    });
  }, []);

  const disabled = useMemo(() => {
    if (loading) {
      return true;
    }
    if (inputMode === "topic") {
      return !topics.some((t) => t.trim());
    }
    if (inputMode === "source") {
      return !text.trim();
    }
    return !file;
  }, [file, inputMode, loading, text, topics]);

  const onSubmit = useCallback(async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const topicList = inputMode === "topic" ? topics.map((t) => t.trim()).filter(Boolean) : [];
      const topicValue = topicList.join(", ");
      const textValue = inputMode === "source" ? text.trim() : "";
      const fileValue = inputMode === "file" ? file : undefined;
      const title = buildMaterialTitle({
        topic: topicValue,
        text: textValue,
        fileName: fileValue?.name ?? "",
      });
      let materialIds: string[] = [];
      if (inputMode === "topic" && topicList.length > 1) {
        const uploads = await Promise.all(
          topicList.map(async (topic) =>
            uploadMaterial({
              subjectTag: topic,
            }),
          ),
        );
        materialIds = uploads.map((row) => row.material_id).filter(Boolean);
      } else {
        const material = await uploadMaterial({
          text: textValue || undefined,
          file: fileValue,
          subjectTag: topicValue || undefined,
        });
        materialIds = [material.material_id];
      }

      const primaryMaterialId = materialIds[0];
      if (!primaryMaterialId) {
        throw new Error("Material creation failed.");
      }

      if (typeof window !== "undefined") {
        const seeds = parseMaterialSeeds(window.localStorage.getItem(MATERIAL_SEEDS_STORAGE_KEY));
        const now = Date.now();
        if (inputMode === "topic" && topicList.length > 1) {
          materialIds.forEach((id, index) => {
            const topic = topicList[index]?.trim();
            if (!id || !topic) {
              return;
            }
            seeds[id] = {
              topic,
              text: undefined,
              title: topic,
              updatedAt: now - index,
            };
          });
        } else {
          seeds[primaryMaterialId] = {
            topic: topicValue || undefined,
            text: textValue ? textValue.slice(0, MAX_SEED_TEXT_CHARS) : undefined,
            title,
            updatedAt: now,
          };
        }
        const ordered = Object.entries(seeds)
          .sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0))
          .slice(0, MAX_MATERIAL_SEEDS);
        window.localStorage.setItem(MATERIAL_SEEDS_STORAGE_KEY, JSON.stringify(Object.fromEntries(ordered)));

        const groups = parseMaterialGroups(window.localStorage.getItem(MATERIAL_GROUPS_STORAGE_KEY));
        if (inputMode === "topic" && topicList.length > 1) {
          groups[primaryMaterialId] = {
            materialIds,
            title,
            updatedAt: now,
          };
        } else {
          delete groups[primaryMaterialId];
        }
        const orderedGroups = Object.entries(groups)
          .sort((a, b) => (b[1].updatedAt || 0) - (a[1].updatedAt || 0))
          .slice(0, MAX_MATERIAL_GROUPS);
        window.localStorage.setItem(MATERIAL_GROUPS_STORAGE_KEY, JSON.stringify(Object.fromEntries(orderedGroups)));
      }
      if (onMaterialCreated) {
        await onMaterialCreated({
          materialId: primaryMaterialId,
          title,
          topic: topicValue || undefined,
          generationMode,
        });
      }
      router.push(`/feed?material_id=${primaryMaterialId}&generation_mode=${generationMode}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something failed");
    } finally {
      setLoading(false);
    }
  }, [file, generationMode, inputMode, onMaterialCreated, router, text, topics]);

  const onFileDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDraggingFile(false);
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) {
      setFile(dropped);
    }
  };

  const reportScrollability = useCallback(() => {
    const element = formRef.current;
    if (!element) {
      return;
    }
    const isScrollable = element.scrollHeight - element.clientHeight > 1;
    onScrollabilityChange?.(isScrollable);
    if (!isScrollable) {
      onScrollOffsetChange?.(false);
    }
  }, [onScrollOffsetChange, onScrollabilityChange]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    reportScrollability();
    const element = formRef.current;
    if (!element) {
      return;
    }
    const onResize = () => {
      reportScrollability();
    };
    window.addEventListener("resize", onResize);
    const observer = typeof ResizeObserver !== "undefined" ? new ResizeObserver(onResize) : null;
    observer?.observe(element);
    return () => {
      window.removeEventListener("resize", onResize);
      observer?.disconnect();
    };
  }, [reportScrollability]);

  useEffect(() => {
    reportScrollability();
  }, [inputMode, reportScrollability, topics.length, text, selectedFileName, error]);

  return (
    <form
      ref={formRef}
      onSubmit={onSubmit}
      onWheelCapture={(event) => {
        const isScrollable = event.currentTarget.scrollHeight - event.currentTarget.clientHeight > 1;
        if (isScrollable && event.deltaY > 0) {
          onScrollGesture?.();
        }
      }}
      onTouchStartCapture={(event) => {
        const nextTouch = event.touches.item(0);
        touchStartYRef.current = nextTouch ? nextTouch.clientY : null;
      }}
      onTouchMoveCapture={(event) => {
        const startY = touchStartYRef.current;
        const nextTouch = event.touches.item(0);
        if (startY === null || !nextTouch) {
          return;
        }
        const isScrollable = event.currentTarget.scrollHeight - event.currentTarget.clientHeight > 1;
        if (isScrollable && startY - nextTouch.clientY > 0) {
          onScrollGesture?.();
        }
      }}
      onTouchEndCapture={() => {
        touchStartYRef.current = null;
      }}
      onScrollCapture={(event) => {
        const isScrollable = event.currentTarget.scrollHeight - event.currentTarget.clientHeight > 1;
        onScrollOffsetChange?.(isScrollable && event.currentTarget.scrollTop > 0);
      }}
      onScroll={(event) => {
        const isScrollable = event.currentTarget.scrollHeight - event.currentTarget.clientHeight > 1;
        onScrollOffsetChange?.(isScrollable && event.currentTarget.scrollTop > 0);
      }}
      className="flex h-full w-full flex-col justify-center overflow-x-visible overflow-y-auto px-6 py-6 md:overflow-hidden md:px-10 md:py-8 lg:px-5"
    >
      <header className="relative mb-4 text-center">
        <img
          src="/logo.png"
          alt="StudyReels logo"
          className="relative z-20 mx-auto hidden h-4 w-[4.75rem] max-w-[26vw] translate-y-16 object-cover opacity-70 md:block"
        />
        <div className="mt-8 md:mt-20">
          <h1 className="relative z-[1] text-[clamp(3.2rem,12vw,8.25rem)] font-black leading-[0.9] tracking-tight text-[#e8e6fc]/30">Study Reels</h1>
          <p className="relative z-20 mt-5 text-sm text-white/68">Pick a mode, add your material, and start your short study feed.</p>
        </div>
      </header>

      <input
        id="material-file"
        className="sr-only"
        type="file"
        accept=".pdf,.docx,.txt"
        onChange={(e) => setFile(e.target.files?.[0])}
      />

      <div className="relative z-20 mt-8 max-w-[300px] md:mt-2 md:max-w-[390px]">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.12em] text-white/70">Input Mode</p>
        <div role="tablist" aria-label="Select input mode" className="relative grid w-full grid-cols-3 rounded-2xl border border-white/25 bg-black/45 p-1">
          <span
            aria-hidden="true"
            className="pointer-events-none absolute bottom-1 left-1 top-1 w-[calc((100%-8px)/3)] rounded-xl bg-white transition-transform duration-300 ease-out"
            style={{
              transform: `translateX(${INPUT_MODE_OPTIONS.findIndex((option) => option.value === inputMode) * 100}%)`,
            }}
          />
          {INPUT_MODE_OPTIONS.map((option) => (
            <button
              key={option.value}
              role="tab"
              type="button"
              aria-selected={inputMode === option.value}
              onClick={() => {
                setInputMode(option.value);
                setError(null);
              }}
              className={`relative z-10 rounded-xl px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-[0.07em] transition-colors md:px-3 md:py-2 md:text-xs ${
                inputMode === option.value ? "text-black" : "text-white/80 hover:text-white"
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div className="relative z-20 mt-6 h-[160px] min-h-[160px] md:mt-4 md:h-[175px] md:min-h-[175px]">
        {inputMode === "topic" ? (
          <>
            <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.12em] text-white/70">Topics</label>
            <div className="h-full min-h-[160px] md:min-h-[175px] flex flex-col gap-2 overflow-y-auto">
              {topics.map((t, i) => (
                <div key={i} className="flex items-center gap-3 pr-1">
                  <div className="w-full rounded-2xl border border-white/30 bg-black/42 backdrop-blur-xl transition focus-within:border-white/65">
                    <input
                      className="h-12 w-full rounded-2xl border-0 bg-transparent px-4 text-sm text-white outline-none placeholder:text-white/40"
                      placeholder={i === 0 ? "e.g. linear regression" : "e.g. another topic"}
                      value={t}
                      onChange={(e) => {
                        const next = [...topics];
                        next[i] = e.target.value;
                        setTopics(next);
                      }}
                    />
                  </div>
                  {topics.length > 1 ? (
                    <button
                      type="button"
                      onClick={() => setTopics(topics.filter((_, j) => j !== i))}
                      className="grid h-8 w-8 shrink-0 place-items-center rounded-lg border border-white/25 bg-black/40 text-white/80 backdrop-blur-xl transition hover:bg-black/55 hover:text-white"
                      aria-label="Remove topic"
                    >
                      <i className="fa-solid fa-xmark text-xs" aria-hidden="true" />
                    </button>
                  ) : null}
                </div>
              ))}
              <button
                type="button"
                onClick={() => setTopics([...topics, ""])}
                className="mt-1 flex w-fit items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-semibold text-white/60 transition hover:text-white/90"
              >
                <i className="fa-solid fa-plus text-[10px]" aria-hidden="true" />
                Add topic
              </button>
            </div>
          </>
        ) : null}

        {inputMode === "source" ? (
          <>
            <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.12em] text-white/70">Text</label>
            <div className="h-full rounded-2xl border border-white/30 bg-black/42 backdrop-blur-xl transition focus-within:border-white/65">
              <textarea
                className="h-full min-h-[160px] w-full resize-none overflow-y-auto rounded-2xl border-0 bg-transparent p-5 text-sm leading-relaxed text-white outline-none placeholder:text-white/40 md:min-h-[175px]"
                placeholder="Paste notes, textbook text, or any material here..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
            </div>
          </>
        ) : null}

        {inputMode === "file" ? (
          <>
            <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.12em] text-white/70">File Upload</label>
            <label
              htmlFor="material-file"
              onDragOver={(event) => {
                event.preventDefault();
                setIsDraggingFile(true);
              }}
              onDragLeave={() => setIsDraggingFile(false)}
              onDrop={onFileDrop}
              className={`flex h-full min-h-[160px] w-full cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed bg-black/42 p-6 text-center text-white outline-none backdrop-blur-xl transition md:min-h-[175px] ${
                isDraggingFile ? "border-white/70 bg-black/52" : "border-white/30"
              }`}
            >
              <span className="grid h-12 w-12 place-items-center rounded-full border border-white/25 bg-black/45 text-white/85">
                <i className="fa-solid fa-arrow-up-from-bracket text-base" aria-hidden="true" />
              </span>
              <p className={`mt-4 max-w-[90%] truncate text-sm font-semibold ${selectedFileName ? "text-white" : "text-white/85"}`}>
                {selectedFileName || "Drag and drop your file here"}
              </p>
              <p className="mt-1 text-xs text-white/58">{selectedFileName ? "Click to replace file" : "Or click to browse (PDF, DOCX, TXT)"}</p>
            </label>
          </>
        ) : null}
      </div>

      <div className="relative z-20 mt-6 shrink-0 flex flex-col gap-2 md:mt-6">
        <p className="min-h-5 text-sm text-white/80">{error ?? ""}</p>
        <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div className="w-full md:max-w-[220px]">
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
                className={`relative z-10 rounded-xl px-2 py-2 text-[11px] font-semibold uppercase tracking-[0.04em] transition-colors duration-200 ${
                  generationMode === "slow" ? "text-black" : "text-white/75 hover:text-white"
                }`}
                aria-pressed={generationMode === "slow"}
              >
                Slow
              </button>
              <button
                type="button"
                onClick={() => setGenerationMode("fast")}
                className={`relative z-10 rounded-xl px-2 py-2 text-[11px] font-semibold uppercase tracking-[0.04em] transition-colors duration-200 ${
                  generationMode === "fast" ? "text-black" : "text-white/75 hover:text-white"
                }`}
                aria-pressed={generationMode === "fast"}
              >
                Fast
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={disabled}
            className="inline-flex w-full items-center justify-center rounded-2xl border border-white/30 bg-white px-7 py-3 text-sm font-bold text-black transition-colors hover:bg-white/92 disabled:cursor-not-allowed disabled:opacity-60 md:w-[12rem]"
          >
            <span className="inline-flex w-[9.5rem] items-center justify-center text-center">
              {loading ? "Starting..." : "Start Learning"}
            </span>
          </button>
        </div>
      </div>
    </form>
  );
}
