"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { HeroGhostSvg } from "@/components/HeroGhostSvg";
import { UploadPanel } from "@/components/UploadPanel";
import { VolumetricLightBackground } from "@/components/VolumetricLightBackground";

const HISTORY_STORAGE_KEY = "studyreels-material-history";
const LEGACY_TOPIC_HISTORY_STORAGE_KEY = "studyreels-reel-topic-history";
const MAX_HISTORY_ITEMS = 120;
const MOBILE_SIDEBAR_CLOSE_MS = 260;

type HistoryItem = {
  materialId: string;
  title: string;
  updatedAt: number;
  starred: boolean;
};

function parseMaterialHistory(raw: string | null): HistoryItem[] {
  if (!raw) {
    return [];
  }
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .filter((item) => item && typeof item.materialId === "string" && typeof item.title === "string")
      .map((item) => ({
        materialId: String(item.materialId),
        title: String(item.title).trim() || "New Study Session",
        updatedAt: Number(item.updatedAt) || 0,
        starred: Boolean(item.starred),
      }))
      .slice(0, MAX_HISTORY_ITEMS);
  } catch {
    return [];
  }
}

function parseLegacyTopicHistory(raw: string | null): HistoryItem[] {
  if (!raw) {
    return [];
  }
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .filter(
        (item) =>
          item &&
          typeof item.materialId === "string" &&
          typeof item.topic === "string" &&
          String(item.materialId).trim() &&
          String(item.topic).trim(),
      )
      .map((item) => ({
        materialId: String(item.materialId),
        title: String(item.topic).trim(),
        updatedAt: Number(item.updatedAt) || 0,
        starred: false,
      }))
      .slice(0, MAX_HISTORY_ITEMS);
  } catch {
    return [];
  }
}

function mergeHistory(primary: HistoryItem[], secondary: HistoryItem[]): HistoryItem[] {
  const map = new Map<string, HistoryItem>();
  for (const item of [...primary, ...secondary]) {
    const existing = map.get(item.materialId);
    if (!existing) {
      map.set(item.materialId, item);
      continue;
    }
    if (item.updatedAt > existing.updatedAt) {
      map.set(item.materialId, { ...item, starred: item.starred || existing.starred });
      continue;
    }
    map.set(item.materialId, { ...existing, starred: existing.starred || item.starred });
  }
  return [...map.values()].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_HISTORY_ITEMS);
}

export default function HomePage() {
  const router = useRouter();
  const [historyQuery, setHistoryQuery] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [mobileSidebarClosing, setMobileSidebarClosing] = useState(false);
  const [activeHistoryMenuId, setActiveHistoryMenuId] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const historyRef = useRef<HistoryItem[]>([]);
  const mobileSidebarCloseTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const base = parseMaterialHistory(window.localStorage.getItem(HISTORY_STORAGE_KEY));
    const legacy = parseLegacyTopicHistory(window.localStorage.getItem(LEGACY_TOPIC_HISTORY_STORAGE_KEY));
    const merged = mergeHistory(base, legacy);
    historyRef.current = merged;
    setHistory(merged);
    window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(merged));
  }, []);

  const historySorted = useMemo(
    () =>
      [...history].sort((a, b) => {
        if (a.starred !== b.starred) {
          return Number(b.starred) - Number(a.starred);
        }
        return b.updatedAt - a.updatedAt;
      }),
    [history],
  );
  const filteredHistory = useMemo(() => {
    const query = historyQuery.trim().toLowerCase();
    if (!query) {
      return historySorted;
    }
    return historySorted.filter((item) => item.title.toLowerCase().includes(query));
  }, [historyQuery, historySorted]);

  const persistHistory = useCallback((next: HistoryItem[]) => {
    historyRef.current = next;
    setHistory(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(next));
    }
  }, []);

  const upsertHistory = useCallback(
    (entry: { materialId: string; title: string; updatedAt: number; starred?: boolean }) => {
      const existing = historyRef.current.find((item) => item.materialId === entry.materialId);
      const merged: HistoryItem = {
        materialId: entry.materialId,
        title: entry.title,
        updatedAt: entry.updatedAt,
        starred: entry.starred ?? existing?.starred ?? false,
      };
      const next = [merged, ...historyRef.current.filter((item) => item.materialId !== merged.materialId)].slice(0, MAX_HISTORY_ITEMS);
      persistHistory(next);
    },
    [persistHistory],
  );

  const clearTopicDraft = useCallback(() => {
    setHistoryQuery("");
    setError(null);
    setActiveHistoryMenuId(null);
  }, []);

  const clearMobileSidebarCloseTimer = useCallback(() => {
    if (mobileSidebarCloseTimerRef.current !== null) {
      window.clearTimeout(mobileSidebarCloseTimerRef.current);
      mobileSidebarCloseTimerRef.current = null;
    }
  }, []);

  const openMobileSidebar = useCallback(() => {
    clearMobileSidebarCloseTimer();
    setMobileSidebarClosing(false);
    setMobileSidebarOpen(true);
  }, [clearMobileSidebarCloseTimer]);

  const closeMobileSidebar = useCallback(() => {
    if (!mobileSidebarOpen || mobileSidebarClosing) {
      return;
    }
    clearMobileSidebarCloseTimer();
    setMobileSidebarClosing(true);
    mobileSidebarCloseTimerRef.current = window.setTimeout(() => {
      setMobileSidebarOpen(false);
      setMobileSidebarClosing(false);
      mobileSidebarCloseTimerRef.current = null;
    }, MOBILE_SIDEBAR_CLOSE_MS);
  }, [clearMobileSidebarCloseTimer, mobileSidebarClosing, mobileSidebarOpen]);

  const forceCloseMobileSidebar = useCallback(() => {
    clearMobileSidebarCloseTimer();
    setMobileSidebarClosing(false);
    setMobileSidebarOpen(false);
  }, [clearMobileSidebarCloseTimer]);

  useEffect(() => {
    return () => {
      clearMobileSidebarCloseTimer();
    };
  }, [clearMobileSidebarCloseTimer]);

  const openMaterialFeed = useCallback(
    (materialId: string) => {
      const existing = historyRef.current.find((item) => item.materialId === materialId);
      if (existing) {
        upsertHistory({ ...existing, updatedAt: Date.now() });
      }
      setActiveHistoryMenuId(null);
      forceCloseMobileSidebar();
      router.push(`/feed?material_id=${materialId}`);
    },
    [forceCloseMobileSidebar, router, upsertHistory],
  );

  const toggleHistoryStar = useCallback(
    (materialId: string) => {
      const existing = historyRef.current.find((item) => item.materialId === materialId);
      if (!existing) {
        return;
      }
      upsertHistory({
        ...existing,
        starred: !existing.starred,
        updatedAt: existing.updatedAt,
      });
      setActiveHistoryMenuId(null);
    },
    [upsertHistory],
  );

  const deleteHistoryItem = useCallback(
    (materialId: string) => {
      const next = historyRef.current.filter((item) => item.materialId !== materialId);
      persistHistory(next);
      setActiveHistoryMenuId((prev) => (prev === materialId ? null : prev));
    },
    [persistHistory],
  );

  useEffect(() => {
    if (!activeHistoryMenuId) {
      return;
    }
    const onPointerDown = (event: PointerEvent) => {
      if (!(event.target instanceof Element)) {
        return;
      }
      if (event.target.closest("[data-history-actions='true']")) {
        return;
      }
      setActiveHistoryMenuId(null);
    };
    window.addEventListener("pointerdown", onPointerDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
    };
  }, [activeHistoryMenuId]);

  const onUploadMaterialCreated = useCallback(
    async (params: { materialId: string; title: string; topic?: string }) => {
      const nextTitle = params.title?.trim() || params.topic?.trim() || "New Study Session";
      upsertHistory({
        materialId: params.materialId,
        title: nextTitle,
        updatedAt: Date.now(),
      });
    },
    [upsertHistory],
  );

  const sidebarPanelContent = (
    <>
      <div className="mt-10 flex items-center justify-end gap-2 lg:mt-0 lg:justify-between">
        <p className="hidden text-lg font-bold leading-none text-white/90 lg:block" aria-label="StudyReels logo">
          ▲
        </p>
        <button
          type="button"
          onClick={clearTopicDraft}
          className="rounded-xl border border-white/25 px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-[0.09em] text-white/90 transition hover:bg-white/12"
        >
          New Chat
        </button>
      </div>

      <div className="mt-3">
        <input
          value={historyQuery}
          onChange={(event) => setHistoryQuery(event.target.value)}
          placeholder="Search history..."
          className="h-9 w-full rounded-xl border border-white/20 bg-black/55 px-3 text-sm text-white outline-none placeholder:text-white/45 focus:border-white/45"
        />
      </div>

      <div className="mt-4 min-h-0 flex-1 overflow-y-auto pr-1">
        <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.1em] text-white/60">History</p>
        <div className="relative pb-1">
          <div className="relative space-y-1.5">
            {filteredHistory.length === 0 ? (
              <p className="text-xs text-white/42">No history yet.</p>
            ) : (
              filteredHistory.map((entry) => (
                <div key={`history-${entry.materialId}`} className="group relative">
                  <button
                    type="button"
                    onClick={() => openMaterialFeed(entry.materialId)}
                    className="h-9 w-full rounded-xl border border-white/15 bg-black/45 px-2.5 pr-10 text-left text-xs text-white/85 transition hover:bg-white/8"
                  >
                    <div className="flex items-center gap-1.5">
                      {entry.starred ? <i className="fa-solid fa-star text-[10px] text-white/75" aria-hidden="true" /> : null}
                      <p className="truncate font-semibold leading-none">{entry.title}</p>
                    </div>
                  </button>

                  <div data-history-actions="true" className="absolute right-1.5 top-1.5 z-20">
                    <button
                      type="button"
                      aria-label="History item actions"
                      onClick={(event) => {
                        event.stopPropagation();
                        setActiveHistoryMenuId((prev) => (prev === entry.materialId ? null : entry.materialId));
                      }}
                      className="grid h-6 w-6 place-items-center rounded-md text-white/70 opacity-100 transition hover:bg-white/10 hover:text-white lg:opacity-0 lg:group-hover:opacity-100 lg:focus-within:opacity-100"
                    >
                      <i className="fa-solid fa-ellipsis text-[11px]" aria-hidden="true" />
                    </button>

                    {activeHistoryMenuId === entry.materialId ? (
                      <div className="absolute right-0 top-full z-30 mt-1 w-36">
                        <div className="absolute inset-0 rounded-xl bg-black/32 backdrop-blur-md" />
                        <div className="relative rounded-xl border border-white/20 bg-black/62 p-1 shadow-lg">
                          <button
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation();
                              toggleHistoryStar(entry.materialId);
                            }}
                            className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-xs text-white/90 transition hover:bg-white/10"
                          >
                            <i
                              className={`fa-${entry.starred ? "solid" : "regular"} fa-star text-[11px] text-white/80`}
                              aria-hidden="true"
                            />
                            {entry.starred ? "Unstar" : "Star"}
                          </button>
                          <button
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation();
                              deleteHistoryItem(entry.materialId);
                            }}
                            className="mt-0.5 flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-xs text-white/90 transition hover:bg-white/10"
                          >
                            <i className="fa-regular fa-trash-can text-[11px] text-white/80" aria-hidden="true" />
                            Delete
                          </button>
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </>
  );

  return (
    <main className="home-hero-shell fixed inset-0 overflow-x-visible overflow-y-auto md:inset-4 md:overflow-hidden">
      <div className="home-hero-bg pointer-events-none absolute inset-0 z-0 overflow-hidden">
        <HeroGhostSvg />
        <div className="absolute -top-[50%] bottom-0 left-0 -right-[100%]">
          <VolumetricLightBackground />
        </div>
      </div>

      {error ? (
        <div className="absolute left-0 right-0 top-3 z-30 mx-auto w-fit rounded-full border border-white/25 bg-black/80 px-4 py-2 text-xs text-white">
          {error}
        </div>
      ) : null}

      <button
        type="button"
        onClick={openMobileSidebar}
        aria-label="Open topic menu"
        style={{
          left: "max(env(safe-area-inset-left), 0px)",
          top: "max(env(safe-area-inset-top), 0px)",
        }}
        className={`fixed z-[70] grid h-10 w-10 place-items-center text-white/90 transition-opacity hover:text-white md:left-7 md:top-7 lg:hidden ${
          mobileSidebarOpen ? "pointer-events-none opacity-0" : "opacity-100"
        }`}
      >
        <i className="fa-solid fa-bars text-base" aria-hidden="true" />
      </button>

      {mobileSidebarOpen ? (
        <div className="fixed left-0 top-0 z-50 h-[100dvh] w-screen lg:hidden">
          <button
            type="button"
            aria-label="Close topic menu overlay"
            onClick={closeMobileSidebar}
            className={`absolute inset-0 bg-black/70 ${mobileSidebarClosing ? "animate-mobile-overlay-out" : "animate-mobile-overlay-in"}`}
          />
          <aside
            className={`absolute left-0 top-0 h-[100dvh] w-[82vw] max-w-[340px] rounded-r-3xl bg-black/42 px-3 pb-3 pt-3 text-white shadow-[0_0_40px_rgba(0,0,0,0.45)] backdrop-blur-xl ${
              mobileSidebarClosing ? "animate-mobile-sidenav-out" : "animate-mobile-sidenav-in"
            }`}
          >
            <p className="absolute left-3 top-4 text-lg font-bold leading-none text-white/90" aria-label="StudyReels logo">
              ▲
            </p>
            <button
              type="button"
              onClick={closeMobileSidebar}
              aria-label="Close topic menu"
              className="absolute right-3 top-1.5 p-1 text-white/80 transition hover:text-white"
            >
              <i className="fa-solid fa-xmark text-base" aria-hidden="true" />
            </button>
            <div className="flex h-full min-h-0 flex-col">{sidebarPanelContent}</div>
          </aside>
        </div>
      ) : null}

      <div className="relative z-10 h-full min-h-0 lg:grid lg:grid-cols-[280px_1px_minmax(0,1fr)]">
        <aside className="hidden min-h-0 flex-col rounded-3xl bg-black/72 px-3 pt-3 pb-2 text-white lg:mt-7 lg:mb-2 lg:flex">
          {sidebarPanelContent}
        </aside>

        <div className="hidden h-full items-center justify-center lg:flex lg:translate-x-3">
          <span className="h-[80%] w-px rounded-full bg-white/20" />
        </div>

        <section className="min-h-0 overflow-visible rounded-3xl bg-black/62 md:overflow-hidden lg:my-2 lg:w-[97%] lg:justify-self-end">
          <UploadPanel onMaterialCreated={onUploadMaterialCreated} />
        </section>
      </div>
    </main>
  );
}
