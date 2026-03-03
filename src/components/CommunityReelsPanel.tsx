"use client";

import { type ChangeEvent, type FormEvent, useCallback, useEffect, useMemo, useState } from "react";

const COMMUNITY_SETS_STORAGE_KEY = "studyreels-community-sets";
const MAX_USER_SETS = 120;
const FALLBACK_THUMBNAIL_URL = "/images/community/ai-systems.svg";
const SUPPORTED_PLATFORMS_LABEL = "YouTube, Instagram, TikTok";
const FEATURED_CAROUSEL_INTERVAL_MS = 5200;
const FEATURED_CAROUSEL_TRANSITION_MS = 520;
const FEATURED_CAROUSEL_PAUSE_MS = 200;

type ReelPlatform = "youtube" | "instagram" | "tiktok";

type CommunityReelEmbed = {
  id: string;
  platform: ReelPlatform;
  sourceUrl: string;
  embedUrl: string;
};

type DraftReelInput = {
  id: string;
  value: string;
};

type ParsedDraftReel = {
  id: string;
  value: string;
  parsed: Omit<CommunityReelEmbed, "id"> | null;
};

type CommunitySet = {
  id: string;
  title: string;
  description: string;
  tags: string[];
  reels: CommunityReelEmbed[];
  reelCount: number;
  curator: string;
  likes: number;
  learners: number;
  updatedLabel: string;
  thumbnailUrl: string;
  featured: boolean;
};

let draftRowCounter = 0;

function createDraftReelRow(value = ""): DraftReelInput {
  draftRowCounter += 1;
  return {
    id: `draft-reel-${draftRowCounter}`,
    value,
  };
}

const PLATFORM_LABEL: Record<ReelPlatform, string> = {
  youtube: "YouTube",
  instagram: "Instagram",
  tiktok: "TikTok",
};

const PLATFORM_ICON: Record<ReelPlatform, string> = {
  youtube: "fa-brands fa-youtube",
  instagram: "fa-brands fa-instagram",
  tiktok: "fa-brands fa-tiktok",
};

const USER_SET_THUMBNAILS = [
  "/images/community/ai-systems.svg",
  "/images/community/language-story.svg",
  "/images/community/civics-debate.svg",
];

const FEATURED_SETS: CommunitySet[] = [
  {
    id: "featured-kinematics-visuals",
    title: "Kinematics Visual Drills",
    description: "Short clips that connect equations of motion to intuitive motion sketches and worked examples.",
    tags: ["physics", "motion", "problem solving"],
    reels: [],
    reelCount: 34,
    curator: "Dr. Ramos",
    likes: 2840,
    learners: 12100,
    updatedLabel: "Updated 2 days ago",
    thumbnailUrl: "/images/community/physics-grid.svg",
    featured: true,
  },
  {
    id: "featured-cell-bio",
    title: "Cell Biology Core",
    description: "A sequence from membrane structure to signaling pathways with high-yield recap reels.",
    tags: ["biology", "cell", "exam prep"],
    reels: [],
    reelCount: 27,
    curator: "MedSchool Crew",
    likes: 1970,
    learners: 8700,
    updatedLabel: "Updated yesterday",
    thumbnailUrl: "/images/community/bio-lab.svg",
    featured: true,
  },
  {
    id: "featured-calc-derivatives",
    title: "Derivatives in Context",
    description: "From slope intuition to optimization and related rates with compact walkthrough reels.",
    tags: ["calculus", "derivatives", "math"],
    reels: [],
    reelCount: 31,
    curator: "Math Forge",
    likes: 2230,
    learners: 9400,
    updatedLabel: "Updated 4 days ago",
    thumbnailUrl: "/images/community/calculus-flow.svg",
    featured: true,
  },
];

const COMMUNITY_LIBRARY_SETS: CommunitySet[] = [
  {
    id: "community-world-history",
    title: "World History in Turning Points",
    description: "Ten key transitions from empire to modern states with source-backed clips.",
    tags: ["history", "timeline"],
    reels: [],
    reelCount: 22,
    curator: "Timeline Lab",
    likes: 960,
    learners: 4100,
    updatedLabel: "Updated 6 days ago",
    thumbnailUrl: "/images/community/civics-debate.svg",
    featured: false,
  },
  {
    id: "community-spanish-conversation",
    title: "Spanish Conversation Starters",
    description: "Pattern-first conversational mini reels for greetings, requests, and follow-ups.",
    tags: ["language", "spanish", "conversation"],
    reels: [],
    reelCount: 18,
    curator: "Lingua Spark",
    likes: 1180,
    learners: 5300,
    updatedLabel: "Updated 3 days ago",
    thumbnailUrl: "/images/community/language-story.svg",
    featured: false,
  },
  {
    id: "community-ml-foundations",
    title: "ML Foundations Fast Track",
    description: "Core probability, model intuition, and overfitting cues in under 20 reels.",
    tags: ["machine learning", "statistics"],
    reels: [],
    reelCount: 19,
    curator: "Data Guild",
    likes: 1450,
    learners: 6200,
    updatedLabel: "Updated 1 week ago",
    thumbnailUrl: "/images/community/ai-systems.svg",
    featured: false,
  },
];

const DEFAULT_COMMUNITY_SETS: CommunitySet[] = [...FEATURED_SETS, ...COMMUNITY_LIBRARY_SETS];

function formatCompact(value: number): string {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1).replace(/\.0$/, "")}K`;
  }
  return String(value);
}

function parseTags(value: string): string[] {
  return Array.from(
    new Set(
      value
        .split(",")
        .map((part) => part.trim().toLowerCase())
        .filter(Boolean),
    ),
  ).slice(0, 6);
}

function toAbsoluteUrl(value: string): URL | null {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  const maybeAbsolute = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
  try {
    return new URL(maybeAbsolute);
  } catch {
    return null;
  }
}

function extractYouTubeVideoId(url: URL): string | null {
  const host = url.hostname.toLowerCase();
  if (host.includes("youtu.be")) {
    const id = url.pathname.split("/").filter(Boolean)[0];
    if (id) {
      return id;
    }
  }
  if (host.includes("youtube.com")) {
    if (url.pathname === "/watch") {
      const id = url.searchParams.get("v");
      if (id) {
        return id;
      }
    }
    if (url.pathname.startsWith("/shorts/")) {
      const id = url.pathname.split("/")[2];
      if (id) {
        return id;
      }
    }
    if (url.pathname.startsWith("/embed/")) {
      const id = url.pathname.split("/")[2];
      if (id) {
        return id;
      }
    }
  }
  return null;
}

function parseReelUrl(input: string): Omit<CommunityReelEmbed, "id"> | null {
  const url = toAbsoluteUrl(input);
  if (!url) {
    return null;
  }
  const host = url.hostname.toLowerCase();

  if (host.includes("youtube.com") || host.includes("youtu.be")) {
    const videoId = extractYouTubeVideoId(url);
    if (!videoId || !/^[A-Za-z0-9_-]{6,}$/.test(videoId)) {
      return null;
    }
    return {
      platform: "youtube",
      sourceUrl: url.toString(),
      embedUrl: `https://www.youtube.com/embed/${videoId}`,
    };
  }

  if (host.includes("instagram.com")) {
    const match = url.pathname.match(/^\/(reel|p|tv)\/([A-Za-z0-9_-]+)/);
    if (!match) {
      return null;
    }
    const kind = match[1];
    const code = match[2];
    return {
      platform: "instagram",
      sourceUrl: url.toString(),
      embedUrl: `https://www.instagram.com/${kind}/${code}/embed`,
    };
  }

  if (host.includes("tiktok.com")) {
    const match = url.pathname.match(/\/video\/(\d+)/);
    if (!match) {
      return null;
    }
    const videoId = match[1];
    return {
      platform: "tiktok",
      sourceUrl: url.toString(),
      embedUrl: `https://www.tiktok.com/embed/v2/${videoId}`,
    };
  }

  return null;
}

function parseStoredReels(raw: unknown): CommunityReelEmbed[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  const parsed: CommunityReelEmbed[] = [];
  for (const [index, entry] of raw.entries()) {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      continue;
    }
    const row = entry as Record<string, unknown>;
    const platform = row.platform;
    const sourceUrl = typeof row.sourceUrl === "string" ? row.sourceUrl.trim() : "";
    const embedUrl = typeof row.embedUrl === "string" ? row.embedUrl.trim() : "";
    if (!sourceUrl || !embedUrl) {
      continue;
    }
    if (platform !== "youtube" && platform !== "instagram" && platform !== "tiktok") {
      continue;
    }
    parsed.push({
      id: typeof row.id === "string" && row.id.trim() ? row.id.trim() : `stored-reel-${index}`,
      platform,
      sourceUrl,
      embedUrl,
    });
  }
  return parsed;
}

function parseStoredSets(raw: string | null): CommunitySet[] {
  if (!raw) {
    return [];
  }
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .filter((item) => item && typeof item === "object" && !Array.isArray(item))
      .map((item) => item as Partial<CommunitySet>)
      .filter(
        (item) =>
          typeof item.id === "string" &&
          Boolean(item.id.trim()) &&
          typeof item.title === "string" &&
          Boolean(item.title.trim()) &&
          typeof item.description === "string",
      )
      .map((item) => {
        const reels = parseStoredReels(item.reels);
        const reelCount = Math.max(reels.length, Math.max(0, Math.floor(Number(item.reelCount) || 0)));
        return {
          id: item.id!.trim(),
          title: item.title!.trim(),
          description: item.description!.trim(),
          tags: Array.isArray(item.tags)
            ? item.tags
                .map((tag) => String(tag || "").trim().toLowerCase())
                .filter(Boolean)
                .slice(0, 6)
            : [],
          reels,
          reelCount,
          curator: typeof item.curator === "string" && item.curator.trim() ? item.curator.trim() : "Community member",
          likes: Math.max(0, Math.floor(Number(item.likes) || 0)),
          learners: Math.max(0, Math.floor(Number(item.learners) || 0)),
          updatedLabel: typeof item.updatedLabel === "string" && item.updatedLabel.trim() ? item.updatedLabel.trim() : "Updated just now",
          thumbnailUrl: typeof item.thumbnailUrl === "string" && item.thumbnailUrl.trim() ? item.thumbnailUrl.trim() : FALLBACK_THUMBNAIL_URL,
          featured: false,
        } as CommunitySet;
      })
      .slice(0, MAX_USER_SETS);
  } catch {
    return [];
  }
}

function getSetReelCount(set: CommunitySet): number {
  return set.reels.length > 0 ? set.reels.length : set.reelCount;
}

function summarizePlatforms(reels: CommunityReelEmbed[]): ReelPlatform[] {
  return Array.from(new Set(reels.map((reel) => reel.platform)));
}

function toTitleCase(value: string): string {
  return value
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((token) => token[0].toUpperCase() + token.slice(1).toLowerCase())
    .join(" ");
}

function getSetIconClass(set: CommunitySet): string {
  const platforms = summarizePlatforms(set.reels);
  if (platforms.includes("youtube")) {
    return "fa-brands fa-youtube";
  }
  if (platforms.includes("instagram")) {
    return "fa-brands fa-instagram";
  }
  if (platforms.includes("tiktok")) {
    return "fa-brands fa-tiktok";
  }
  return "fa-solid fa-layer-group";
}

type CommunityReelsPanelMode = "community" | "create";

type CommunityReelsPanelProps = {
  mode?: CommunityReelsPanelMode;
};

type FeaturedTransitionStage = "idle" | "exiting" | "pause" | "entering";

export function CommunityReelsPanel({ mode = "community" }: CommunityReelsPanelProps) {
  const [activeCommunityCategory, setActiveCommunityCategory] = useState("Featured");
  const [activeFeaturedIndex, setActiveFeaturedIndex] = useState(0);
  const [leavingFeaturedIndex, setLeavingFeaturedIndex] = useState<number | null>(null);
  const [pendingFeaturedIndex, setPendingFeaturedIndex] = useState<number | null>(null);
  const [featuredTransitionStage, setFeaturedTransitionStage] = useState<FeaturedTransitionStage>("idle");
  const [featuredTransitionDirection, setFeaturedTransitionDirection] = useState<1 | -1>(1);
  const [selectedDirectorySet, setSelectedDirectorySet] = useState<CommunitySet | null>(null);
  const [isViewingSelectedSet, setIsViewingSelectedSet] = useState(false);
  const [query, setQuery] = useState("");
  const [setTitle, setSetTitle] = useState("");
  const [setDescription, setSetDescription] = useState("");
  const [setTags, setSetTags] = useState("");
  const [thumbnailPreview, setThumbnailPreview] = useState("");
  const [reelInputs, setReelInputs] = useState<DraftReelInput[]>(() => [createDraftReelRow()]);
  const [createError, setCreateError] = useState<string | null>(null);
  const [createSuccess, setCreateSuccess] = useState<string | null>(null);
  const [userSets, setUserSets] = useState<CommunitySet[]>([]);
  const [storageHydrated, setStorageHydrated] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    setUserSets(parseStoredSets(window.localStorage.getItem(COMMUNITY_SETS_STORAGE_KEY)));
    setStorageHydrated(true);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || !storageHydrated) {
      return;
    }
    window.localStorage.setItem(COMMUNITY_SETS_STORAGE_KEY, JSON.stringify(userSets.slice(0, MAX_USER_SETS)));
  }, [storageHydrated, userSets]);

  const allSets = useMemo(() => [...userSets, ...DEFAULT_COMMUNITY_SETS], [userSets]);

  const featuredSets = useMemo(() => DEFAULT_COMMUNITY_SETS.filter((set) => set.featured), []);

  const filteredSets = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return allSets;
    }
    return allSets.filter((set) => {
      if (set.title.toLowerCase().includes(normalized)) {
        return true;
      }
      if (set.description.toLowerCase().includes(normalized)) {
        return true;
      }
      if (set.curator.toLowerCase().includes(normalized)) {
        return true;
      }
      if (set.tags.some((tag) => tag.includes(normalized))) {
        return true;
      }
      return set.reels.some((reel) => PLATFORM_LABEL[reel.platform].toLowerCase().includes(normalized));
    });
  }, [allSets, query]);

  const communityCategories = useMemo(() => {
    const tagCounts = new Map<string, number>();
    for (const set of allSets) {
      for (const tag of set.tags) {
        const normalized = tag.trim().toLowerCase();
        if (!normalized) {
          continue;
        }
        tagCounts.set(normalized, (tagCounts.get(normalized) ?? 0) + 1);
      }
    }
    const topTags = [...tagCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([tag]) => toTitleCase(tag))
      .filter(Boolean)
      .slice(0, 3);
    return ["Featured", ...topTags];
  }, [allSets]);

  useEffect(() => {
    if (!communityCategories.includes(activeCommunityCategory)) {
      setActiveCommunityCategory(communityCategories[0] ?? "Featured");
    }
  }, [activeCommunityCategory, communityCategories]);

  const categoryFilteredSets = useMemo(() => {
    if (activeCommunityCategory === "Featured") {
      const featuredFirst = filteredSets.filter((set) => set.featured);
      const others = filteredSets.filter((set) => !set.featured);
      return [...featuredFirst, ...others];
    }
    const key = activeCommunityCategory.trim().toLowerCase();
    const matched = filteredSets.filter((set) => {
      if (set.title.toLowerCase().includes(key) || set.description.toLowerCase().includes(key)) {
        return true;
      }
      return set.tags.some((tag) => tag.toLowerCase().includes(key));
    });
    return matched.length > 0 ? matched : filteredSets;
  }, [activeCommunityCategory, filteredSets]);

  const fallbackHeroSet = useMemo(() => categoryFilteredSets[0] ?? filteredSets[0] ?? null, [categoryFilteredSets, filteredSets]);

  const featuredCarouselSets = useMemo(() => {
    if (featuredSets.length > 0) {
      return featuredSets.slice(0, 3);
    }
    return fallbackHeroSet ? [fallbackHeroSet] : [];
  }, [fallbackHeroSet, featuredSets]);

  useEffect(() => {
    if (featuredCarouselSets.length === 0) {
      setActiveFeaturedIndex(0);
      setLeavingFeaturedIndex(null);
      setPendingFeaturedIndex(null);
      setFeaturedTransitionStage("idle");
      return;
    }
    setActiveFeaturedIndex((prev) => {
      if (prev < featuredCarouselSets.length) {
        return prev;
      }
      return 0;
    });
    setLeavingFeaturedIndex(null);
    setPendingFeaturedIndex(null);
    setFeaturedTransitionStage("idle");
  }, [featuredCarouselSets.length]);

  const startFeaturedTransition = useCallback((nextIndex: number) => {
    if (featuredCarouselSets.length <= 1 || featuredTransitionStage !== "idle") {
      return;
    }
    const setCount = featuredCarouselSets.length;
    const normalized = ((nextIndex % setCount) + setCount) % setCount;
    if (normalized === activeFeaturedIndex) {
      return;
    }
    const forwardSteps = (normalized - activeFeaturedIndex + setCount) % setCount;
    const backwardSteps = (activeFeaturedIndex - normalized + setCount) % setCount;
    setFeaturedTransitionDirection(forwardSteps <= backwardSteps ? 1 : -1);
    setLeavingFeaturedIndex(activeFeaturedIndex);
    setPendingFeaturedIndex(normalized);
    setFeaturedTransitionStage("exiting");
  }, [activeFeaturedIndex, featuredCarouselSets.length, featuredTransitionStage]);

  const goToFeaturedSet = useCallback(
    (nextIndex: number) => {
      startFeaturedTransition(nextIndex);
    },
    [startFeaturedTransition],
  );

  const goToNextFeaturedSet = useCallback(() => {
    startFeaturedTransition(activeFeaturedIndex + 1);
  }, [activeFeaturedIndex, startFeaturedTransition]);

  useEffect(() => {
    if (mode !== "community" || featuredCarouselSets.length <= 1) {
      return;
    }
    const intervalId = window.setInterval(() => {
      goToNextFeaturedSet();
    }, FEATURED_CAROUSEL_INTERVAL_MS);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [featuredCarouselSets.length, goToNextFeaturedSet, mode]);

  useEffect(() => {
    if (featuredTransitionStage !== "exiting") {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      setLeavingFeaturedIndex(null);
      setFeaturedTransitionStage("pause");
    }, FEATURED_CAROUSEL_TRANSITION_MS);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [featuredTransitionStage]);

  useEffect(() => {
    if (featuredTransitionStage !== "pause") {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      if (pendingFeaturedIndex === null) {
        setFeaturedTransitionStage("idle");
        return;
      }
      setActiveFeaturedIndex(pendingFeaturedIndex);
      setPendingFeaturedIndex(null);
      setFeaturedTransitionStage("entering");
    }, FEATURED_CAROUSEL_PAUSE_MS);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [featuredTransitionStage, pendingFeaturedIndex]);

  useEffect(() => {
    if (featuredTransitionStage !== "entering") {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      setFeaturedTransitionStage("idle");
    }, FEATURED_CAROUSEL_TRANSITION_MS);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [featuredTransitionStage]);

  useEffect(() => {
    if (!selectedDirectorySet) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setSelectedDirectorySet(null);
        setIsViewingSelectedSet(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [selectedDirectorySet]);

  const activeFeaturedSet = featuredCarouselSets[activeFeaturedIndex] ?? fallbackHeroSet;

  const directorySets = useMemo(() => {
    if (!activeFeaturedSet) {
      return categoryFilteredSets;
    }
    return categoryFilteredSets.filter((set) => set.id !== activeFeaturedSet.id);
  }, [activeFeaturedSet, categoryFilteredSets]);

  const parsedDraftReels = useMemo<ParsedDraftReel[]>(
    () =>
      reelInputs.map((row) => {
        const trimmed = row.value.trim();
        if (!trimmed) {
          return { id: row.id, value: row.value, parsed: null };
        }
        return {
          id: row.id,
          value: row.value,
          parsed: parseReelUrl(trimmed),
        };
      }),
    [reelInputs],
  );

  const validDraftReelCount = useMemo(
    () => parsedDraftReels.filter((row) => row.value.trim() && row.parsed !== null).length,
    [parsedDraftReels],
  );

  const onThumbnailFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    if (!file.type.startsWith("image/")) {
      setCreateError("Thumbnail must be an image file.");
      setCreateSuccess(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        setThumbnailPreview(reader.result);
        setCreateError(null);
      }
    };
    reader.readAsDataURL(file);
  };

  const onCreateSet = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const title = setTitle.trim();
    const description = setDescription.trim();
    if (!title) {
      setCreateError("Set name is required.");
      setCreateSuccess(null);
      return;
    }
    if (description.length < 18) {
      setCreateError("Add a short description so others know what this set covers.");
      setCreateSuccess(null);
      return;
    }
    if (!thumbnailPreview) {
      setCreateError("Add a thumbnail image before posting.");
      setCreateSuccess(null);
      return;
    }

    const nonEmptyRows = parsedDraftReels.filter((row) => row.value.trim());
    if (nonEmptyRows.length === 0) {
      setCreateError("Add at least one reel URL to post this set.");
      setCreateSuccess(null);
      return;
    }
    const firstInvalid = nonEmptyRows.find((row) => row.parsed === null);
    if (firstInvalid) {
      setCreateError(`One or more reel links are invalid. Supported: ${SUPPORTED_PLATFORMS_LABEL}.`);
      setCreateSuccess(null);
      return;
    }

    const now = Date.now();
    const parsedReels = nonEmptyRows.map((row, index) => {
      const parsed = row.parsed!;
      return {
        id: `user-reel-${now}-${index}`,
        platform: parsed.platform,
        sourceUrl: parsed.sourceUrl,
        embedUrl: parsed.embedUrl,
      } as CommunityReelEmbed;
    });

    const tags = parseTags(setTags);
    const createdSet: CommunitySet = {
      id: `user-set-${now}`,
      title,
      description,
      tags,
      reels: parsedReels,
      reelCount: parsedReels.length,
      curator: "You",
      likes: 0,
      learners: 1,
      updatedLabel: "Updated just now",
      thumbnailUrl: thumbnailPreview,
      featured: false,
    };

    setUserSets((prev) => [createdSet, ...prev].slice(0, MAX_USER_SETS));
    setSetTitle("");
    setSetDescription("");
    setSetTags("");
    setThumbnailPreview("");
    setReelInputs([createDraftReelRow()]);
    setCreateError(null);
    setCreateSuccess(`"${title}" posted with ${parsedReels.length} reels.`);
  };

  const addReelInputRow = () => {
    setReelInputs((prev) => [...prev, createDraftReelRow()]);
  };

  const removeReelInputRow = (rowId: string) => {
    setReelInputs((prev) => {
      const next = prev.filter((row) => row.id !== rowId);
      return next.length > 0 ? next : [createDraftReelRow()];
    });
  };

  const updateReelInputRow = (rowId: string, value: string) => {
    setReelInputs((prev) => prev.map((row) => (row.id === rowId ? { ...row, value } : row)));
  };

  const openDirectorySet = (set: CommunitySet) => {
    setSelectedDirectorySet(set);
    setIsViewingSelectedSet(false);
  };

  const closeDirectorySetModal = () => {
    setSelectedDirectorySet(null);
    setIsViewingSelectedSet(false);
  };

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden px-3 pb-5 pt-14 text-white sm:px-5 sm:pb-6 md:px-7 md:pb-7 md:pt-20 lg:px-8 lg:py-7">
      <div className="shrink-0">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between md:gap-4">
          <div className="w-full pl-5 sm:pl-6 md:w-auto md:pl-8 lg:pl-3">
            <div className="flex items-center justify-center gap-2 md:justify-start">
              <h2 className="text-xl font-semibold tracking-tight text-white sm:text-2xl md:text-[1.9rem]">Community Sets</h2>
              <span className="rounded-full border border-white/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.09em] text-white/55">Beta</span>
            </div>
          </div>
          {mode === "community" ? (
            <label className="mx-auto block w-[calc(100%-0.25rem)] self-stretch md:mx-0 md:mr-5 md:w-[20.5rem] md:self-auto lg:mr-3 lg:w-[23rem]">
              <div className="relative">
                <i className="fa-solid fa-magnifying-glass pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-sm text-white/45" />
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search community sets"
                  className="h-11 w-full rounded-xl border border-white/20 bg-black/35 pl-11 pr-4 text-sm text-white outline-none placeholder:text-white/45 focus:border-white/40 sm:h-12 sm:pl-12"
                />
              </div>
            </label>
          ) : (
            <p className="max-w-lg text-xs text-white/62">Build a set with unlimited reels and post it to the community feed.</p>
          )}
        </div>
      </div>

      <div className="mt-3 min-h-0 flex-1 overflow-hidden md:mt-4">
        {mode === "community" ? (
          <div className="balanced-scroll-gutter h-full min-h-0 space-y-4 overflow-y-auto md:space-y-5">
            {featuredCarouselSets.length > 0 ? (
              <section className="group/featured relative min-h-[360px] overflow-hidden rounded-[1.5rem] border border-white/15 p-4 pb-12 max-[380px]:pb-10 sm:min-h-[430px] sm:rounded-[2rem] sm:p-5 sm:pb-14 md:min-h-[500px] md:p-7 md:pb-16 lg:min-h-[520px] lg:p-8">
                <div
                  className="pointer-events-none absolute inset-0 opacity-95"
                  style={{
                    background:
                      "radial-gradient(circle at 18% 28%, rgba(27, 133, 255, 0.72), transparent 45%), radial-gradient(circle at 84% 38%, rgba(182, 232, 255, 0.62), transparent 54%), linear-gradient(118deg, #0b5f98 0%, #0782cf 44%, #96c7d6 100%)",
                  }}
                />
                <div className="pointer-events-none absolute inset-0 bg-black/20" />

                {featuredCarouselSets.length > 1 ? (
                  <>
                    <button
                      type="button"
                      aria-label="Next featured set"
                      onClick={goToNextFeaturedSet}
                      className="absolute right-4 top-4 z-30 grid h-9 w-9 place-items-center rounded-full bg-black/45 text-white/82 transition-all duration-200 hover:bg-black/60 hover:text-white md:pointer-events-none md:right-6 md:top-1/2 md:-translate-y-1/2 md:opacity-0 md:group-hover/featured:pointer-events-auto md:group-hover/featured:opacity-100 md:group-focus-within/featured:pointer-events-auto md:group-focus-within/featured:opacity-100"
                    >
                      <i className="fa-solid fa-chevron-right text-[10px]" aria-hidden="true" />
                    </button>
                  </>
                ) : null}

                <div className="relative z-10 min-h-[285px] overflow-hidden sm:min-h-[320px] md:min-h-[410px]">
                  {featuredCarouselSets.map((set, index) => {
                    const isLeaving = featuredTransitionStage === "exiting" && index === leavingFeaturedIndex;
                    const isActive = index === activeFeaturedIndex && featuredTransitionStage !== "exiting" && featuredTransitionStage !== "pause";
                    if (!isActive && !isLeaving) {
                      return null;
                    }
                    const motionClass = isLeaving
                      ? featuredTransitionDirection === 1
                        ? "animate-featured-slide-exit-forward"
                        : "animate-featured-slide-exit-backward"
                      : featuredTransitionStage === "entering"
                        ? featuredTransitionDirection === 1
                          ? "animate-featured-slide-enter-forward"
                          : "animate-featured-slide-enter-backward"
                        : "opacity-100";
                    return (
                      <article
                        key={`${set.id}-${isLeaving ? "leaving" : "active"}`}
                        className={`absolute inset-0 ${isLeaving ? "z-10 pointer-events-none" : "z-20"} ${motionClass}`}
                      >
                        <div className="grid min-h-[285px] gap-5 max-[380px]:gap-4 sm:min-h-[320px] sm:gap-7 md:min-h-[410px] md:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] md:items-center">
                          <div className="flex max-w-2xl flex-col items-start text-left md:pl-2 lg:pl-8">
                            <p className="inline-flex rounded-full bg-white/12 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.1em] text-white/88">
                              Featured Set
                            </p>
                            <h3 className="mt-5 text-[1.65rem] font-semibold leading-[1.12] text-white max-[380px]:mt-3 sm:mt-4 sm:text-[2rem] md:text-[2.6rem] lg:text-[3.05rem]">{set.title}</h3>
                            <p className="mt-5 max-w-xl text-[0.95rem] leading-relaxed text-white/84 max-[380px]:mt-3 sm:mt-4 sm:text-[1.02rem] md:text-[1.08rem] lg:text-[1.18rem]">{set.description}</p>

                            <div className="mt-7 flex flex-wrap items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/78 max-[380px]:mt-5 sm:mt-5 sm:gap-2.5 sm:text-[11px]">
                              <span className="rounded-full bg-black/30 px-2 py-1">{getSetReelCount(set)} reels</span>
                              <span className="rounded-full bg-black/30 px-2 py-1">{formatCompact(set.learners)} learners</span>
                              <span className="rounded-full bg-black/30 px-2 py-1">{formatCompact(set.likes)} likes</span>
                            </div>

                            <button
                              type="button"
                              className="mt-8 inline-flex w-full items-center justify-center rounded-full bg-white px-6 py-2.5 text-sm font-semibold text-black transition max-[380px]:mt-6 hover:bg-[#f1eee5] sm:mt-7 sm:w-auto sm:px-8 sm:py-3"
                            >
                              View Set
                            </button>
                          </div>

                          <div className="relative hidden h-full min-h-[320px] md:block">
                            <div className="absolute right-8 top-2 z-20 max-w-[78%] rounded-full border border-white/35 bg-white/20 px-3 py-1.5 text-xs font-semibold text-white/92 backdrop-blur-xl lg:right-16 lg:max-w-[72%] lg:px-4 lg:py-2 lg:text-sm">
                              @{set.curator} trending this week
                            </div>
                            <div className="absolute bottom-0 right-8 w-[84%] overflow-hidden rounded-[1.4rem] border border-white/25 bg-black/30 lg:right-12 lg:w-[80%] lg:rounded-[1.7rem]">
                              <img
                                src={set.thumbnailUrl || FALLBACK_THUMBNAIL_URL}
                                alt={`${set.title} cover`}
                                className="h-[280px] w-full object-contain md:h-[310px] lg:h-[330px]"
                              />
                            </div>
                          </div>
                        </div>
                      </article>
                    );
                  })}
                </div>

                {featuredCarouselSets.length > 1 ? (
                  <div className="absolute bottom-4 left-1/2 z-20 flex -translate-x-1/2 items-center justify-center gap-2 md:bottom-5">
                    {featuredCarouselSets.map((set, index) => (
                      <button
                        key={`featured-dot-${set.id}`}
                        type="button"
                        aria-label={`Go to featured set ${index + 1}`}
                        onClick={() => goToFeaturedSet(index)}
                        className={`h-2 rounded-full transition-all ${
                          index === activeFeaturedIndex ? "w-6 bg-white" : "w-2 bg-white/45 hover:bg-white/70"
                        }`}
                      />
                    ))}
                  </div>
                ) : null}
              </section>
            ) : null}

            <section className="flex items-center gap-2 overflow-x-auto pb-1">
              {communityCategories.map((category) => (
                <button
                  key={category}
                  type="button"
                  onClick={() => setActiveCommunityCategory(category)}
                  className={`whitespace-nowrap rounded-full px-3 py-1.5 text-xs font-medium transition-all duration-200 ease-out sm:px-4 sm:py-2 sm:text-sm ${
                    activeCommunityCategory === category
                      ? "bg-white text-black"
                      : "bg-black/30 text-white/75 hover:bg-white/10 hover:text-white"
                  }`}
                >
                  {category}
                </button>
              ))}
            </section>

            <section>
              <div className="mb-2 flex items-center justify-between">
                <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-white/60">Community Directory</p>
                <p className="text-[10px] font-semibold uppercase tracking-[0.09em] text-white/45">{directorySets.length} sets</p>
              </div>
              {directorySets.length === 0 ? (
                <p className="rounded-xl bg-black/40 px-3 py-4 text-sm text-white/65">No sets matched that search.</p>
              ) : (
                <div className="grid gap-2.5 md:grid-cols-2 md:gap-x-4 md:gap-y-3 lg:gap-x-10">
                  {directorySets.map((set) => {
                    const reelCount = getSetReelCount(set);
                    return (
                      <button
                        type="button"
                        key={set.id}
                        onClick={() => openDirectorySet(set)}
                        className="group relative flex w-full items-center gap-2.5 rounded-xl bg-black/20 px-3 py-3 text-left backdrop-blur-md transition-all duration-200 ease-out hover:bg-white/10 sm:gap-3 sm:rounded-2xl"
                      >
                        <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-black/30 text-white/82 transition-colors duration-200 group-hover:text-white sm:h-10 sm:w-10 sm:rounded-xl">
                          <i className="fa-regular fa-square text-sm sm:text-base" aria-hidden="true" />
                        </span>
                        <div className="min-w-0 flex-1 text-left">
                          <p className="w-full truncate text-[0.97rem] font-medium text-white transition-colors duration-200 group-hover:text-white sm:text-[1.02rem]">{set.title}</p>
                          <p className="mt-0.5 hidden w-full truncate text-sm text-white/58 transition-colors duration-200 group-hover:text-white/78 lg:block">{set.description}</p>
                          <span className="mt-1 inline-flex rounded-full bg-white/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/62 transition-colors duration-200 group-hover:text-white/80">
                            {reelCount} reels
                          </span>
                        </div>
                        <span
                          className="grid h-7 w-7 shrink-0 place-items-center self-center rounded-full text-white/58 transition-colors duration-200 group-hover:text-white/80 sm:h-8 sm:w-8"
                          aria-hidden="true"
                        >
                          <i className="fa-solid fa-chevron-right text-[11px]" />
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </section>
          </div>
        ) : (
          <div className="balanced-scroll-gutter h-full min-h-0 overflow-y-auto">
            <section className="rounded-3xl p-2 sm:p-4 md:p-6">
              <div className="mb-5">
                <p className="text-[11px] font-semibold uppercase tracking-[0.11em] text-white/68">Create A Set</p>
              </div>

              <form onSubmit={onCreateSet} className="space-y-5 md:space-y-6">
                <div className="space-y-5">
                  <label className="block">
                    <span className="mb-2 block text-xs text-white/72">Set Name</span>
                    <input
                      value={setTitle}
                      onChange={(event) => setSetTitle(event.target.value)}
                      placeholder="Example: Organic Chemistry Reactions"
                      className="h-11 w-full rounded-xl border border-white/15 bg-black/55 px-3 text-sm text-white outline-none placeholder:text-white/40 focus:border-white/45"
                    />
                  </label>

                  <label className="block">
                    <span className="mb-2 block text-xs text-white/72">Description</span>
                    <textarea
                      value={setDescription}
                      onChange={(event) => setSetDescription(event.target.value)}
                      placeholder="What does this set cover and who is it for?"
                      className="h-24 w-full resize-none rounded-xl border border-white/15 bg-black/55 px-3 py-2 text-sm text-white outline-none placeholder:text-white/40 focus:border-white/45 md:h-20"
                    />
                  </label>

                  <label className="block">
                    <span className="mb-2 block text-xs text-white/72">Tags</span>
                    <input
                      value={setTags}
                      onChange={(event) => setSetTags(event.target.value)}
                      placeholder="chemistry, reaction mechanisms, exam prep"
                      className="h-11 w-full rounded-xl border border-white/15 bg-black/55 px-3 text-sm text-white outline-none placeholder:text-white/40 focus:border-white/45"
                    />
                  </label>

                  <div>
                    <p className="mb-2 block text-xs text-white/72">Thumbnail Image</p>
                    <input id="community-set-thumbnail" type="file" accept="image/*" className="sr-only" onChange={onThumbnailFileChange} />
                    <div className="flex flex-col items-start gap-3 sm:flex-row sm:gap-4">
                      <label
                        htmlFor="community-set-thumbnail"
                        className="group relative block h-[120px] w-full max-w-[220px] cursor-pointer overflow-hidden rounded-xl border border-dashed border-white/15 bg-black/55 sm:h-[90px] sm:w-[140px] sm:max-w-none"
                      >
                        <img
                          src={thumbnailPreview || USER_SET_THUMBNAILS[0] || FALLBACK_THUMBNAIL_URL}
                          alt="Set thumbnail preview"
                          className={`h-full w-full object-cover transition ${thumbnailPreview ? "opacity-100" : "opacity-55 group-hover:opacity-75"}`}
                        />
                        <span className="absolute inset-0 grid place-items-center bg-black/35 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/85">
                          {thumbnailPreview ? "Change" : "Upload"}
                        </span>
                      </label>
                      <div className="w-full space-y-2 sm:w-auto">
                        <p className="text-[11px] text-white/62">Use a vertical image for better mobile previews.</p>
                        {thumbnailPreview ? (
                          <button
                            type="button"
                            onClick={() => setThumbnailPreview("")}
                            className="inline-flex items-center gap-1 rounded-lg border border-white/15 bg-black/45 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/72 transition hover:bg-white/10 hover:text-white"
                          >
                            <i className="fa-solid fa-trash text-[9px]" aria-hidden="true" />
                            Remove
                          </button>
                        ) : null}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="min-h-0 rounded-2xl p-3.5">
                  <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.09em] text-white/68">Embed Reels ({SUPPORTED_PLATFORMS_LABEL})</p>
                    <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-white/55">{validDraftReelCount} valid</p>
                  </div>
                  <div className="balanced-scroll-gutter max-h-[300px] space-y-3 overflow-y-auto">
                    {parsedDraftReels.map((row, index) => {
                      const hasInput = Boolean(row.value.trim());
                      const hasValidEmbed = row.parsed !== null;
                      return (
                        <div key={row.id} className="rounded-xl p-3">
                          <div className="flex flex-col items-stretch gap-2 sm:flex-row sm:items-center sm:gap-3">
                            <input
                              value={row.value}
                              onChange={(event) => updateReelInputRow(row.id, event.target.value)}
                              placeholder="Paste YouTube, Instagram reel, or TikTok URL"
                              className="h-10 w-full rounded-lg border border-white/15 bg-black/60 px-2.5 text-xs text-white outline-none placeholder:text-white/40 focus:border-white/45 sm:h-9"
                            />
                            <button
                              type="button"
                              onClick={() => removeReelInputRow(row.id)}
                              className="inline-flex h-9 w-full shrink-0 items-center justify-center gap-1 rounded-lg border border-white/15 bg-black/55 text-white/72 transition hover:bg-white/10 hover:text-white sm:grid sm:h-8 sm:w-8 sm:place-items-center"
                              aria-label={`Remove reel input ${index + 1}`}
                            >
                              <i className="fa-solid fa-xmark text-xs" aria-hidden="true" />
                              <span className="text-[10px] font-semibold uppercase tracking-[0.08em] sm:hidden">Remove</span>
                            </button>
                          </div>

                          {hasInput && !hasValidEmbed ? (
                            <p className="mt-2 text-[11px] text-[#ffb4b4]">Invalid URL. Supported: {SUPPORTED_PLATFORMS_LABEL}.</p>
                          ) : null}

                          {row.parsed ? (
                            <div className="mt-3 overflow-hidden rounded-lg">
                              <div className="flex items-center gap-1.5 px-2 py-1.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/70">
                                <i className={PLATFORM_ICON[row.parsed.platform]} aria-hidden="true" />
                                {PLATFORM_LABEL[row.parsed.platform]} embed
                              </div>
                              <iframe
                                src={row.parsed.embedUrl}
                                title={`${PLATFORM_LABEL[row.parsed.platform]} reel preview`}
                                className="h-[180px] w-full border-0 sm:h-[160px]"
                                loading="lazy"
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                                allowFullScreen
                              />
                            </div>
                          ) : null}
                        </div>
                      );
                    })}
                  </div>
                </div>

                <button
                  type="button"
                  onClick={addReelInputRow}
                  className="inline-flex -mt-2 items-center gap-1 self-start px-0 py-1.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/72 transition hover:text-white"
                >
                  <i className="fa-solid fa-plus text-[9px]" aria-hidden="true" />
                  Add Reels
                </button>

                {createError ? <p className="text-xs text-[#ffb4b4]">{createError}</p> : null}
                {createSuccess ? <p className="text-xs text-[#9ef8cb]">{createSuccess}</p> : null}

                <button
                  type="submit"
                  className="inline-flex h-11 w-full items-center justify-center rounded-xl border border-white/15 bg-black/55 px-4 text-sm font-semibold text-white transition hover:bg-white hover:text-black"
                >
                  Post Community Set
                </button>
              </form>
            </section>
          </div>
        )}
      </div>

      {selectedDirectorySet ? (
        <div className="fixed inset-0 z-[120] flex items-end justify-center p-0 sm:p-4 md:items-center md:p-8">
          <button type="button" aria-label="Close set details" onClick={closeDirectorySetModal} className="absolute inset-0 bg-black/82" />
          <section
            role="dialog"
            aria-modal="true"
            aria-label={`${selectedDirectorySet.title} details`}
            className="relative z-10 flex max-h-[92dvh] w-full max-w-3xl flex-col overflow-hidden rounded-t-3xl border border-white/18 bg-black/96 sm:max-h-[88vh] sm:rounded-3xl"
          >
            <div className="flex items-start justify-between gap-4 px-4 pb-4 pt-5 sm:px-5 md:px-6 md:pt-6">
              <div className="min-w-0">
                <p className="text-[10px] font-semibold uppercase tracking-[0.1em] text-white/58">Community Set</p>
                <h3 className="mt-1 truncate text-lg font-semibold text-white sm:text-xl md:text-2xl">{selectedDirectorySet.title}</h3>
              </div>
              <button
                type="button"
                onClick={closeDirectorySetModal}
                aria-label="Close set details"
                className="grid h-8 w-8 place-items-center rounded-lg border border-white/20 text-white/78 transition hover:text-white"
              >
                <i className="fa-solid fa-xmark text-sm" aria-hidden="true" />
              </button>
            </div>

            {!isViewingSelectedSet ? (
              <div className="balanced-scroll-gutter min-h-0 space-y-4 overflow-y-auto px-4 pb-2 sm:px-5 md:px-6">
                <div className="grid gap-4 md:grid-cols-[180px_minmax(0,1fr)]">
                  <div className="h-[180px] overflow-hidden rounded-2xl border border-white/15 bg-black/45 md:h-auto">
                    <img src={selectedDirectorySet.thumbnailUrl || FALLBACK_THUMBNAIL_URL} alt={`${selectedDirectorySet.title} thumbnail`} className="h-full w-full object-cover" />
                  </div>
                  <div className="space-y-3">
                    <p className="text-sm leading-relaxed text-white/82">{selectedDirectorySet.description}</p>
                    <div className="flex flex-wrap gap-2 text-[11px] text-white/72">
                      <span className="rounded-full bg-white/10 px-2.5 py-1">{getSetReelCount(selectedDirectorySet)} reels</span>
                      <span className="rounded-full bg-white/10 px-2.5 py-1">{formatCompact(selectedDirectorySet.learners)} learners</span>
                      <span className="rounded-full bg-white/10 px-2.5 py-1">{formatCompact(selectedDirectorySet.likes)} likes</span>
                      <span className="rounded-full bg-white/10 px-2.5 py-1">Curated by {selectedDirectorySet.curator}</span>
                    </div>
                    <p className="text-xs text-white/56">{selectedDirectorySet.updatedLabel}</p>
                  </div>
                </div>

                {selectedDirectorySet.tags.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {selectedDirectorySet.tags.map((tag) => (
                      <span key={`${selectedDirectorySet.id}-tag-${tag}`} className="rounded-full bg-black/55 px-2.5 py-1 text-[11px] text-white/68">
                        #{tag}
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="balanced-scroll-gutter min-h-0 space-y-3 overflow-y-auto px-4 pb-2 sm:px-5 md:px-6">
                {selectedDirectorySet.reels.length === 0 ? (
                  <p className="rounded-2xl bg-white/6 px-4 py-5 text-sm text-white/68">No embedded reels were added to this set yet.</p>
                ) : (
                  selectedDirectorySet.reels.map((reel) => (
                    <article key={reel.id} className="overflow-hidden rounded-2xl border border-white/12 bg-black/55">
                      <div className="flex items-center gap-2 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/70">
                        <i className={PLATFORM_ICON[reel.platform]} aria-hidden="true" />
                        {PLATFORM_LABEL[reel.platform]}
                      </div>
                      <iframe
                        src={reel.embedUrl}
                        title={`${selectedDirectorySet.title} reel`}
                        className="h-[200px] w-full border-0 sm:h-[220px]"
                        loading="lazy"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                        allowFullScreen
                      />
                    </article>
                  ))
                )}
              </div>
            )}

            <div className="mt-3 flex flex-col-reverse gap-2 border-t border-white/10 px-4 py-4 sm:flex-row sm:items-center sm:justify-end sm:px-5 md:px-6">
              {isViewingSelectedSet ? (
                <button
                  type="button"
                  onClick={() => setIsViewingSelectedSet(false)}
                  className="inline-flex h-10 w-full items-center justify-center rounded-xl border border-white/20 px-4 text-xs font-semibold uppercase tracking-[0.08em] text-white/85 transition hover:text-white sm:w-auto"
                >
                  Back To Details
                </button>
              ) : null}
              <button
                type="button"
                onClick={() => setIsViewingSelectedSet(true)}
                className="inline-flex h-10 w-full items-center justify-center rounded-xl border border-white/20 bg-white px-4 text-xs font-semibold uppercase tracking-[0.08em] text-black transition hover:bg-[#f1eee5] sm:w-auto"
              >
                View This Reel Set
              </button>
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
}
