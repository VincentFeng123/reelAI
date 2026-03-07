"use client";

import { type ChangeEvent, type DragEvent, type FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { createCommunitySet, fetchCommunitySets } from "@/lib/api";

const COMMUNITY_SETS_STORAGE_KEY = "studyreels-community-sets";
const MAX_USER_SETS = 120;
const FALLBACK_THUMBNAIL_URL = "/images/community/ai-systems.svg";
const SUPPORTED_PLATFORMS_LABEL = "YouTube, Instagram, TikTok";
const MIN_SET_DESCRIPTION_LENGTH = 18;
const FEATURED_CAROUSEL_INTERVAL_MS = 5200;
const FEATURED_CAROUSEL_TRANSITION_MS = 520;
const FEATURED_CAROUSEL_PAUSE_MS = 200;
const FEATURED_CAROUSEL_CONTENT_MIN_HEIGHT_FALLBACK = 410;
const FEATURED_CAROUSEL_CONTENT_MIN_HEIGHT_TOUCH_FALLBACK = 280;
const FEATURED_CAROUSEL_BUTTON_BOTTOM_MARGIN_PX = 18;
const FEATURED_CAROUSEL_IMAGE_BOTTOM_MARGIN_PX = 18;
const DIRECTORY_DETAIL_TRANSITION_MS = 440;
const DETAIL_CONTENT_TOP_PADDING_FALLBACK = 420;
const DETAIL_CONTENT_TOP_PADDING_GUTTER = 16;
const DETAIL_CONTENT_TOP_PADDING_UPSHIFT_PX = 56;
const DETAIL_CAROUSEL_VISIBLE_COUNT = 3;
const DETAIL_BANNER_LEFT_EXPANSION_PX = 10;

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

const DETAIL_LOREM_PARAGRAPHS = [
  "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer luctus, lorem ut porta vehicula, lectus lectus viverra mi, id faucibus turpis est eget augue.",
  "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Velit euismod in pellentesque massa placerat duis ultricies lacus sed turpis tincidunt id aliquet.",
  "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Egestas integer eget aliquet nibh praesent tristique magna sit.",
  "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Cras semper auctor neque vitae tempus quam pellentesque nec nam aliquam.",
  "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Convallis aenean et tortor at risus viverra adipiscing at in tellus.",
  "Nunc consequat interdum varius sit amet mattis vulputate enim nulla aliquet porttitor lacus luctus accumsan tortor posuere ac ut consequat semper viverra nam libero justo.",
  "Purus gravida quis blandit turpis cursus in hac habitasse platea dictumst quisque sagittis purus sit amet volutpat consequat mauris nunc congue nisi vitae suscipit tellus.",
  "Aliquet sagittis id consectetur purus ut faucibus pulvinar elementum integer enim neque volutpat ac tincidunt vitae semper quis lectus nulla at volutpat diam ut venenatis.",
  "Pharetra pharetra massa massa ultricies mi quis hendrerit dolor magna eget est lorem ipsum dolor sit amet consectetur adipiscing elit pellentesque habitant morbi tristique.",
  "Amet dictum sit amet justo donec enim diam vulputate ut pharetra sit amet aliquam id diam maecenas ultricies mi eget mauris pharetra et ultrices neque ornare aenean.",
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
  isVisible?: boolean;
  onDetailOpenChange?: (isOpen: boolean) => void;
};

type FeaturedTransitionStage = "idle" | "exiting" | "pause" | "entering";

export function CommunityReelsPanel({ mode = "community", isVisible = true, onDetailOpenChange }: CommunityReelsPanelProps) {
  const [activeCommunityCategory, setActiveCommunityCategory] = useState("Featured");
  const [activeFeaturedIndex, setActiveFeaturedIndex] = useState(0);
  const [leavingFeaturedIndex, setLeavingFeaturedIndex] = useState<number | null>(null);
  const [pendingFeaturedIndex, setPendingFeaturedIndex] = useState<number | null>(null);
  const [featuredTransitionStage, setFeaturedTransitionStage] = useState<FeaturedTransitionStage>("idle");
  const [featuredTransitionDirection, setFeaturedTransitionDirection] = useState<1 | -1>(1);
  const [featuredCarouselContentHeight, setFeaturedCarouselContentHeight] = useState(FEATURED_CAROUSEL_CONTENT_MIN_HEIGHT_FALLBACK);
  const [selectedDirectorySet, setSelectedDirectorySet] = useState<CommunitySet | null>(null);
  const [isDirectoryDetailOpen, setIsDirectoryDetailOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [setTitle, setSetTitle] = useState("");
  const [setDescription, setSetDescription] = useState("");
  const [setTags, setSetTags] = useState("");
  const [thumbnailPreview, setThumbnailPreview] = useState("");
  const [detailCarouselIndex, setDetailCarouselIndex] = useState(0);
  const [reelInputs, setReelInputs] = useState<DraftReelInput[]>(() => [createDraftReelRow()]);
  const [createError, setCreateError] = useState<string | null>(null);
  const [createSuccess, setCreateSuccess] = useState<string | null>(null);
  const [isPostingSet, setIsPostingSet] = useState(false);
  const [userSets, setUserSets] = useState<CommunitySet[]>([]);
  const [storageHydrated, setStorageHydrated] = useState(false);
  const [portalReady, setPortalReady] = useState(false);
  const [detailBannerLeft, setDetailBannerLeft] = useState(0);
  const [detailBannerHeight, setDetailBannerHeight] = useState(0);
  const [isDetailBannerCompact, setIsDetailBannerCompact] = useState(false);
  const [isThumbnailDragOver, setIsThumbnailDragOver] = useState(false);
  const [thumbnailFileName, setThumbnailFileName] = useState("");
  const panelRootRef = useRef<HTMLDivElement | null>(null);
  const detailBannerRef = useRef<HTMLDivElement | null>(null);
  const detailContentScrollRef = useRef<HTMLDivElement | null>(null);
  const communityScrollRef = useRef<HTMLDivElement | null>(null);
  const activeFeaturedSlideRef = useRef<HTMLDivElement | null>(null);
  const directoryDetailCloseTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    let cancelled = false;
    setPortalReady(true);
    const localSets = parseStoredSets(window.localStorage.getItem(COMMUNITY_SETS_STORAGE_KEY));
    setUserSets(localSets);
    setStorageHydrated(true);
    void (async () => {
      try {
        const remoteSets = await fetchCommunitySets();
        if (cancelled) {
          return;
        }
        setUserSets(remoteSets.slice(0, MAX_USER_SETS));
      } catch {
        // Keep local cache fallback if backend is unavailable.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || !storageHydrated) {
      return;
    }
    window.localStorage.setItem(COMMUNITY_SETS_STORAGE_KEY, JSON.stringify(userSets.slice(0, MAX_USER_SETS)));
  }, [storageHydrated, userSets]);

  const clearDirectoryDetailCloseTimer = useCallback(() => {
    if (directoryDetailCloseTimerRef.current !== null) {
      window.clearTimeout(directoryDetailCloseTimerRef.current);
      directoryDetailCloseTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      clearDirectoryDetailCloseTimer();
    };
  }, [clearDirectoryDetailCloseTimer]);

  const updateDetailBannerGeometry = useCallback(() => {
    if (!panelRootRef.current) {
      return;
    }
    const panelRect = panelRootRef.current.getBoundingClientRect();
    const nextLeft = Math.max(0, Math.round(panelRect.left) - DETAIL_BANNER_LEFT_EXPANSION_PX);
    setDetailBannerLeft((prev) => (prev === nextLeft ? prev : nextLeft));

    if (!detailBannerRef.current) {
      return;
    }
    const nextHeight = Math.round(detailBannerRef.current.getBoundingClientRect().height);
    setDetailBannerHeight((prev) => (prev === nextHeight ? prev : nextHeight));
  }, []);

  const allSets = useMemo(() => [...userSets, ...DEFAULT_COMMUNITY_SETS], [userSets]);
  const featuredCarouselSets = useMemo(() => FEATURED_SETS.slice(0, 3), []);
  const detailCarouselImages = useMemo(() => {
    if (!selectedDirectorySet) {
      return [];
    }
    const images = Array.from(
      new Set([selectedDirectorySet.thumbnailUrl, ...DEFAULT_COMMUNITY_SETS.map((set) => set.thumbnailUrl), FALLBACK_THUMBNAIL_URL].filter(Boolean)),
    );
    while (images.length < DETAIL_CAROUSEL_VISIBLE_COUNT) {
      images.push(FALLBACK_THUMBNAIL_URL);
    }
    return images;
  }, [selectedDirectorySet]);

  const filteredDirectorySets = useMemo(() => {
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
    const hasQuery = query.trim().length > 0;
    if (hasQuery) {
      return filteredDirectorySets;
    }

    if (activeCommunityCategory === "Featured") {
      const featuredFirst = filteredDirectorySets.filter((set) => set.featured);
      const others = filteredDirectorySets.filter((set) => !set.featured);
      return [...featuredFirst, ...others];
    }

    const normalizedCategory = activeCommunityCategory.trim().toLowerCase();
    return filteredDirectorySets.filter((set) => {
      if (set.title.toLowerCase().includes(normalizedCategory)) {
        return true;
      }
      if (set.description.toLowerCase().includes(normalizedCategory)) {
        return true;
      }
      return set.tags.some((tag) => tag.toLowerCase().includes(normalizedCategory));
    });
  }, [activeCommunityCategory, filteredDirectorySets, query]);

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

  const isSearchActive = query.trim().length > 0;

  useEffect(() => {
    if (mode !== "community" || isSearchActive || featuredCarouselSets.length === 0) {
      return;
    }
    const measure = () => {
      const activeSlide = activeFeaturedSlideRef.current;
      if (!activeSlide) {
        return;
      }
      const slideRect = activeSlide.getBoundingClientRect();
      const imageTarget = activeSlide.querySelector<HTMLElement>("[data-featured-image-target]");
      const ctaButton = activeSlide.querySelector<HTMLButtonElement>("[data-featured-view-set-button]");
      const imageBottom =
        imageTarget && imageTarget.offsetParent !== null
          ? imageTarget.getBoundingClientRect().bottom - slideRect.top + FEATURED_CAROUSEL_IMAGE_BOTTOM_MARGIN_PX
          : 0;
      const buttonBottom =
        ctaButton && ctaButton.offsetParent !== null
          ? ctaButton.getBoundingClientRect().bottom - slideRect.top + FEATURED_CAROUSEL_BUTTON_BOTTOM_MARGIN_PX
          : 0;
      const measuredHeight = Math.ceil(activeSlide.scrollHeight);
      const nextHeight = imageBottom > 0
        ? Math.max(FEATURED_CAROUSEL_CONTENT_MIN_HEIGHT_FALLBACK, Math.ceil(imageBottom))
        : buttonBottom > 0
          ? Math.max(FEATURED_CAROUSEL_CONTENT_MIN_HEIGHT_TOUCH_FALLBACK, Math.ceil(buttonBottom))
          : Math.max(FEATURED_CAROUSEL_CONTENT_MIN_HEIGHT_FALLBACK, measuredHeight);
      setFeaturedCarouselContentHeight((prev) => (prev === nextHeight ? prev : nextHeight));
    };
    measure();
    const rafId = window.requestAnimationFrame(measure);
    window.addEventListener("resize", measure);

    const resizeObserver = typeof ResizeObserver !== "undefined" && activeFeaturedSlideRef.current ? new ResizeObserver(measure) : null;
    if (resizeObserver && activeFeaturedSlideRef.current) {
      resizeObserver.observe(activeFeaturedSlideRef.current);
    }

    return () => {
      window.cancelAnimationFrame(rafId);
      window.removeEventListener("resize", measure);
      resizeObserver?.disconnect();
    };
  }, [activeFeaturedIndex, featuredCarouselSets.length, featuredTransitionStage, isSearchActive, mode]);

  const directorySets = categoryFilteredSets;
  const maxDetailCarouselIndex = Math.max(0, detailCarouselImages.length - DETAIL_CAROUSEL_VISIBLE_COUNT);

  const goToPreviousDetailCarousel = useCallback(() => {
    setDetailCarouselIndex((prev) => Math.max(0, prev - 1));
  }, []);

  const goToNextDetailCarousel = useCallback(() => {
    setDetailCarouselIndex((prev) => Math.min(maxDetailCarouselIndex, prev + 1));
  }, [maxDetailCarouselIndex]);

  const onCommunityCategoryChange = useCallback((category: string) => {
    if (category === activeCommunityCategory) {
      return;
    }
    const previousScrollTop = communityScrollRef.current?.scrollTop ?? 0;
    setActiveCommunityCategory(category);
    if (typeof window === "undefined") {
      return;
    }
    window.requestAnimationFrame(() => {
      const container = communityScrollRef.current;
      if (!container) {
        return;
      }
      container.scrollTop = Math.min(previousScrollTop, container.scrollHeight - container.clientHeight);
    });
  }, [activeCommunityCategory]);

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
  const nonEmptyDraftReelCount = useMemo(
    () => parsedDraftReels.filter((row) => row.value.trim()).length,
    [parsedDraftReels],
  );
  const invalidDraftReelCount = Math.max(0, nonEmptyDraftReelCount - validDraftReelCount);
  const normalizedSetTitle = setTitle.trim();
  const normalizedSetDescription = setDescription.trim();
  const descriptionCharsRemaining = Math.max(0, MIN_SET_DESCRIPTION_LENGTH - normalizedSetDescription.length);
  const descriptionHasTooFewChars = normalizedSetDescription.length > 0 && descriptionCharsRemaining > 0;
  const parsedSetTags = useMemo(() => parseTags(setTags), [setTags]);
  const requiredCompletionCount =
    (normalizedSetTitle ? 1 : 0) +
    (normalizedSetDescription.length >= MIN_SET_DESCRIPTION_LENGTH ? 1 : 0) +
    (thumbnailPreview ? 1 : 0) +
    (validDraftReelCount > 0 && invalidDraftReelCount === 0 ? 1 : 0);
  const completionPercent = Math.round((requiredCompletionCount / 4) * 100);
  const canPostSet = requiredCompletionCount === 4 && !isPostingSet;

  const applyThumbnailFile = useCallback((file: File | null | undefined) => {
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
        setThumbnailFileName(file.name);
        setCreateError(null);
      }
    };
    reader.readAsDataURL(file);
  }, []);

  const onThumbnailFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    applyThumbnailFile(event.target.files?.[0]);
    event.target.value = "";
  };

  const onThumbnailDragEnter = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsThumbnailDragOver(true);
  };

  const onThumbnailDragOver = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = "copy";
    setIsThumbnailDragOver(true);
  };

  const onThumbnailDragLeave = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const nextTarget = event.relatedTarget as Node | null;
    if (nextTarget && event.currentTarget.contains(nextTarget)) {
      return;
    }
    setIsThumbnailDragOver(false);
  };

  const onThumbnailDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsThumbnailDragOver(false);
    applyThumbnailFile(event.dataTransfer.files?.[0]);
  };

  const onCreateSet = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const title = setTitle.trim();
    const description = setDescription.trim();
    setCreateSuccess(null);
    if (!title) {
      setCreateError("Set name is required.");
      setCreateSuccess(null);
      return;
    }
    if (description.length < MIN_SET_DESCRIPTION_LENGTH) {
      setCreateError(`Description must be at least ${MIN_SET_DESCRIPTION_LENGTH} characters.`);
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

    const parsedReels = nonEmptyRows.map((row, index) => {
      const parsed = row.parsed!;
      return {
        platform: parsed.platform,
        sourceUrl: parsed.sourceUrl,
        embedUrl: parsed.embedUrl,
      };
    });

    const tags = parseTags(setTags);
    setIsPostingSet(true);
    try {
      const createdSet = await createCommunitySet({
        title,
        description,
        tags,
        reels: parsedReels,
        thumbnailUrl: thumbnailPreview,
        curator: "You",
      });
      setUserSets((prev) => [createdSet, ...prev.filter((item) => item.id !== createdSet.id)].slice(0, MAX_USER_SETS));
      setSetTitle("");
      setSetDescription("");
      setSetTags("");
      setThumbnailPreview("");
      setThumbnailFileName("");
      setReelInputs([createDraftReelRow()]);
      setCreateError(null);
      setCreateSuccess(`"${createdSet.title}" posted with ${createdSet.reels.length} reels.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not post community set.";
      setCreateError(message);
    } finally {
      setIsPostingSet(false);
    }
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

  const closeDirectorySetModal = useCallback(() => {
    if (!selectedDirectorySet) {
      return;
    }
    clearDirectoryDetailCloseTimer();
    setIsDetailBannerCompact(false);
    setIsDirectoryDetailOpen(false);
    directoryDetailCloseTimerRef.current = window.setTimeout(() => {
      setSelectedDirectorySet(null);
      directoryDetailCloseTimerRef.current = null;
    }, DIRECTORY_DETAIL_TRANSITION_MS);
  }, [clearDirectoryDetailCloseTimer, selectedDirectorySet]);

  const openDirectorySet = useCallback(
    (set: CommunitySet) => {
      clearDirectoryDetailCloseTimer();
      updateDetailBannerGeometry();
      setSelectedDirectorySet(set);
      setIsDetailBannerCompact(false);
      if (typeof window === "undefined") {
        setIsDirectoryDetailOpen(true);
        return;
      }
      window.requestAnimationFrame(() => {
        setIsDirectoryDetailOpen(true);
      });
    },
    [clearDirectoryDetailCloseTimer, updateDetailBannerGeometry],
  );

  useEffect(() => {
    if (!isDirectoryDetailOpen) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeDirectorySetModal();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [closeDirectorySetModal, isDirectoryDetailOpen]);

  useEffect(() => {
    if (!onDetailOpenChange) {
      return;
    }
    onDetailOpenChange(mode === "community" && isVisible && isDirectoryDetailOpen);
  }, [isDirectoryDetailOpen, isVisible, mode, onDetailOpenChange]);

  useEffect(() => {
    if (!selectedDirectorySet) {
      setIsDetailBannerCompact(false);
    }
  }, [selectedDirectorySet]);

  useEffect(() => {
    setDetailCarouselIndex(0);
  }, [selectedDirectorySet?.id]);

  useEffect(() => {
    setDetailCarouselIndex((prev) => Math.min(prev, maxDetailCarouselIndex));
  }, [maxDetailCarouselIndex]);

  useEffect(() => {
    if (mode !== "community" || !isDirectoryDetailOpen || !selectedDirectorySet) {
      return;
    }
    const el = detailContentScrollRef.current;
    if (!el) {
      return;
    }
    const onScroll = () => {
      setIsDetailBannerCompact((prev) => {
        const next = el.scrollTop > 0;
        return prev === next ? prev : next;
      });
    };
    onScroll();
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      el.removeEventListener("scroll", onScroll);
    };
  }, [isDirectoryDetailOpen, mode, selectedDirectorySet]);

  useEffect(() => {
    if (mode !== "community" || !portalReady || !selectedDirectorySet) {
      return;
    }
    const update = () => {
      updateDetailBannerGeometry();
    };
    update();
    const rafId = window.requestAnimationFrame(update);
    window.addEventListener("resize", update);

    const resizeObserver = typeof ResizeObserver !== "undefined" ? new ResizeObserver(update) : null;
    if (resizeObserver && panelRootRef.current) {
      resizeObserver.observe(panelRootRef.current);
    }
    if (resizeObserver && detailBannerRef.current) {
      resizeObserver.observe(detailBannerRef.current);
    }

    return () => {
      window.cancelAnimationFrame(rafId);
      window.removeEventListener("resize", update);
      resizeObserver?.disconnect();
    };
  }, [isDirectoryDetailOpen, mode, portalReady, selectedDirectorySet, updateDetailBannerGeometry]);

  const detailContentTopPadding = Math.max(
    DETAIL_CONTENT_TOP_PADDING_FALLBACK,
    detailBannerHeight + DETAIL_CONTENT_TOP_PADDING_GUTTER - DETAIL_CONTENT_TOP_PADDING_UPSHIFT_PX,
  );

  const detailBannerPortal =
    mode === "community" && isVisible && portalReady && selectedDirectorySet
      ? createPortal(
        <div
          ref={detailBannerRef}
          className={`pointer-events-none fixed top-0 z-[96] overflow-hidden bg-transparent ${
            isDetailBannerCompact ? "backdrop-blur-[10px]" : "backdrop-blur-[4px]"
          } transition-[opacity,left,backdrop-filter] duration-[560ms] ease-[cubic-bezier(0.25,0.1,0.25,1)] ${
            isDirectoryDetailOpen ? "opacity-100" : "opacity-0 pointer-events-none"
          }`}
            style={{ left: `${detailBannerLeft + 3}px`, right: 0 }}
          >
            <div
              className={`pointer-events-none absolute inset-0 bg-white/[0.04] transition-opacity duration-[420ms] ease-out ${
                isDetailBannerCompact ? "opacity-0" : "opacity-100"
              }`}
            />
            <div
              className={`relative z-10 transition-[padding] duration-[840ms] ease-[cubic-bezier(0.2,0.85,0.25,1)] ${
                isDetailBannerCompact
                  ? "px-4 pt-4 sm:px-6 sm:pt-4 md:px-7"
                  : "px-4 pt-[calc(max(env(safe-area-inset-top),0px)+24px)] sm:px-6 sm:pt-[calc(max(env(safe-area-inset-top),0px)+28px)] md:px-7 md:pt-[calc(max(env(safe-area-inset-top),0px)+34px)]"
              }`}
            >
              <div
                className={`overflow-hidden will-change-[max-height,transform,opacity] transition-[max-height,opacity,transform,padding] duration-[840ms] ease-[cubic-bezier(0.2,0.85,0.25,1)] ${
                  isDetailBannerCompact ? "max-h-0 -translate-y-3 opacity-0 pb-0" : "max-h-[920px] translate-y-0 opacity-100 pb-12 sm:pb-14 md:pb-16"
                }`}
              >
                <button
                  type="button"
                  onClick={closeDirectorySetModal}
                  className="pointer-events-auto mt-1 inline-flex items-center gap-2 rounded-full px-2.5 py-2 text-[13px] font-semibold text-white/90 transition hover:text-white sm:text-sm"
                >
                  <i className="fa-solid fa-chevron-left text-[11px] sm:text-xs" aria-hidden="true" />
                  Community Sets
                </button>

                <div className="mt-12 flex flex-col gap-5 sm:mt-14 sm:gap-6 md:mt-16">
                  <span className="grid h-12 w-12 place-items-center rounded-2xl bg-black/28 text-white/90 sm:h-14 sm:w-14">
                    <i className={`${getSetIconClass(selectedDirectorySet)} text-lg sm:text-xl`} aria-hidden="true" />
                  </span>

                  <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                    <div className="min-w-0">
                      <p className="text-[10px] font-semibold uppercase tracking-[0.1em] text-white/70">Community Set</p>
                      <h3 className="mt-1 text-[1.9rem] font-semibold leading-[1.08] text-white sm:text-[2.2rem] md:text-[2.8rem]">{selectedDirectorySet.title}</h3>
                      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-white/85 sm:text-[0.98rem] md:text-[1.05rem]">{selectedDirectorySet.description}</p>
                    </div>

                    <button
                      type="button"
                      className="pointer-events-auto inline-flex h-10 items-center justify-center self-start rounded-full bg-white px-5 text-sm font-semibold text-[#06233a] transition hover:bg-[#d9eefb] md:self-center"
                    >
                      View Reels
                    </button>
                  </div>
                </div>

                <div className="mt-4 flex flex-wrap items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/80 sm:mt-5 sm:text-[11px]">
                  <span className="rounded-full bg-black/28 px-2.5 py-1">{getSetReelCount(selectedDirectorySet)} reels</span>
                  <span className="rounded-full bg-black/28 px-2.5 py-1">{formatCompact(selectedDirectorySet.learners)} learners</span>
                  <span className="rounded-full bg-black/28 px-2.5 py-1">{formatCompact(selectedDirectorySet.likes)} likes</span>
                  <span className="rounded-full bg-black/28 px-2.5 py-1">Curated by {selectedDirectorySet.curator}</span>
                </div>
              </div>

              <div
                className={`overflow-hidden transition-[max-height,opacity,transform,backdrop-filter] duration-[640ms] ease-[cubic-bezier(0.2,0.85,0.25,1)] ${
                  isDetailBannerCompact
                    ? "pointer-events-auto h-16 max-h-16 translate-y-0 rounded-2xl border border-white/20 bg-white/[0.03] opacity-100 backdrop-blur-[10px] sm:h-20 sm:max-h-20"
                    : "pointer-events-none h-0 max-h-0 -translate-y-2 opacity-0"
                }`}
              >
                <div className="mx-auto flex h-full w-full max-w-none items-center justify-between gap-3 px-3 sm:px-4">
                  <div className="ml-1 min-w-0 max-w-[62%] flex items-center gap-2.5 sm:ml-2">
                    <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-black/28 text-white/90">
                      <i className={`${getSetIconClass(selectedDirectorySet)} text-sm`} aria-hidden="true" />
                    </span>
                    <p className="truncate text-[0.96rem] font-semibold text-white sm:text-[1.02rem]">{selectedDirectorySet.title}</p>
                  </div>
                  <button
                    type="button"
                    className="pointer-events-auto mr-2 inline-flex h-9 shrink-0 items-center justify-center rounded-full bg-white px-4 text-xs font-semibold text-[#06233a] transition hover:bg-[#d9eefb] sm:mr-3 sm:px-5"
                  >
                    View Reels
                  </button>
                </div>
              </div>
            </div>
          </div>,
          document.body,
        )
      : null;

  return (
    <div
      ref={panelRootRef}
      className="flex h-full min-h-0 flex-col overflow-hidden px-3 pb-5 pt-14 text-white sm:px-5 sm:pb-6 md:px-7 md:pb-7 md:pt-20 lg:px-8 lg:py-7"
    >
      {mode === "community" ? (
        <div className="relative min-h-0 flex-1 overflow-hidden">
          <div
            className={`absolute inset-0 flex min-h-0 flex-col transition-opacity duration-[440ms] ease-[cubic-bezier(0.22,1,0.36,1)] ${
              isDirectoryDetailOpen ? "opacity-0 pointer-events-none" : "opacity-100"
            }`}
            aria-hidden={isDirectoryDetailOpen}
          >
            <div className="shrink-0">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between md:gap-4">
                <div className="w-full pl-5 sm:pl-6 md:w-auto md:pl-8 lg:pl-3">
                  <div className="flex items-center justify-center gap-2 md:justify-start">
                    <h2 className="text-xl font-semibold tracking-tight text-white sm:text-2xl md:text-[1.9rem]">Community Sets</h2>
                    <span className="rounded-full border border-white/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.09em] text-white/55">Beta</span>
                  </div>
                </div>
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
              </div>
            </div>

            <div className="mt-3 min-h-0 flex-1 overflow-hidden md:mt-4">
              <div ref={communityScrollRef} className="balanced-scroll-gutter h-full min-h-0 space-y-4 overflow-y-auto md:space-y-5">
            {!isSearchActive && featuredCarouselSets.length > 0 ? (
              <section className="group/featured relative overflow-hidden rounded-[1.5rem] border border-[#2b2b2b] bg-transparent p-4 pb-12 shadow-[inset_0_1px_0_rgba(255,255,255,0.12)] backdrop-blur-[4px] max-[380px]:pb-10 sm:rounded-[2rem] sm:p-5 sm:pb-14 md:p-7 md:pb-16 lg:p-8">
                <div className="pointer-events-none absolute inset-0 bg-white/[0.04]" />

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

                <div className="relative z-10 overflow-hidden" style={{ minHeight: `${featuredCarouselContentHeight}px` }}>
                  {featuredCarouselSets.map((set, index) => {
                    const isLeaving = featuredTransitionStage === "exiting" && index === leavingFeaturedIndex;
                    const isActive = index === activeFeaturedIndex && featuredTransitionStage !== "exiting" && featuredTransitionStage !== "pause";
                    if (!isActive && !isLeaving) {
                      return null;
                    }
                    const motionClass = isLeaving
                      ? featuredTransitionDirection === 1
                        ? "animate-featured-fade-exit animate-featured-slide-exit-forward"
                        : "animate-featured-fade-exit animate-featured-slide-exit-backward"
                      : featuredTransitionStage === "entering"
                        ? featuredTransitionDirection === 1
                          ? "animate-featured-fade-enter animate-featured-slide-enter-forward"
                          : "animate-featured-fade-enter animate-featured-slide-enter-backward"
                        : "opacity-100";
                    return (
                      <article
                        key={`${set.id}-${isLeaving ? "leaving" : "active"}`}
                        className={`absolute inset-0 ${isLeaving ? "z-10 pointer-events-none" : "z-20"} ${motionClass}`}
                      >
                        <div
                          ref={isLeaving ? null : activeFeaturedSlideRef}
                          className="grid min-h-[285px] gap-5 max-[380px]:gap-4 sm:min-h-[320px] sm:gap-7 md:min-h-[410px] md:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] md:items-center"
                        >
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
                              data-featured-view-set-button
                              className="mt-8 inline-flex w-full items-center justify-center rounded-full bg-white px-6 py-2.5 text-sm font-semibold text-black transition max-[380px]:mt-6 hover:bg-[#f1eee5] sm:mt-7 sm:w-auto sm:px-8 sm:py-3"
                            >
                              View Set
                            </button>
                          </div>

                          <div className="relative hidden h-full min-h-[320px] md:block">
                            <div className="absolute right-8 top-2 z-20 max-w-[78%] rounded-full border border-white/35 bg-white/20 px-3 py-1.5 text-xs font-semibold text-white/92 backdrop-blur-xl lg:right-16 lg:max-w-[72%] lg:px-4 lg:py-2 lg:text-sm">
                              @{set.curator} trending this week
                            </div>
                            <div
                              data-featured-image-target
                              className="absolute bottom-0 right-8 w-[84%] overflow-hidden rounded-[1.4rem] border border-white/25 bg-black/30 lg:right-12 lg:w-[80%] lg:rounded-[1.7rem]"
                            >
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

            {!isSearchActive ? (
            <section className="flex items-center gap-2 overflow-x-auto pb-1">
              {communityCategories.map((category) => (
                <button
                  key={category}
                  type="button"
                  onClick={() => onCommunityCategoryChange(category)}
                  className={`whitespace-nowrap rounded-full px-3 py-1.5 text-xs font-medium transition-all duration-200 ease-out sm:px-4 sm:py-2 sm:text-sm ${
                    activeCommunityCategory === category
                      ? "bg-white text-black"
                      : "bg-black/30 text-white/75 hover:bg-white/20 hover:text-white hover:backdrop-blur-sm"
                  }`}
                >
                  {category}
                </button>
              ))}
            </section>
            ) : null}

            <section className="relative overflow-hidden rounded-2xl bg-transparent px-3 py-3 backdrop-blur-[3px] sm:px-4 sm:py-3.5">
              <div className="pointer-events-none absolute inset-0 bg-white/[0.04]" />
              <div className="relative z-10">
              <div className="mb-2 flex items-center justify-between">
                <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-white/60">
                  {isSearchActive ? "Search Results" : "Community Directory"}
                </p>
                <p className="text-[10px] font-semibold uppercase tracking-[0.09em] text-white/45">{directorySets.length} sets</p>
              </div>
              {directorySets.length === 0 ? (
                <p className="px-3 py-4 text-sm text-white/65">
                  {isSearchActive ? "No sets matched your search." : "No sets matched that search."}
                </p>
              ) : (
                <div className={isSearchActive ? "flex flex-col gap-2.5" : "grid gap-2.5 md:grid-cols-2 md:gap-x-4 md:gap-y-3 lg:gap-x-10"}>
                  {directorySets.map((set) => {
                    const reelCount = getSetReelCount(set);
                    return (
                      <button
                        type="button"
                        key={set.id}
                        onClick={() => openDirectorySet(set)}
                        className="group relative flex w-full items-center gap-2.5 rounded-xl bg-[#1c1c1c] px-3 py-3 text-left transition-all duration-200 ease-out hover:bg-[#121212] sm:gap-3 sm:rounded-2xl"
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
              </div>
            </section>
              </div>
            </div>
          </div>

          <section
            role="dialog"
            aria-modal="true"
            aria-label={selectedDirectorySet ? `${selectedDirectorySet.title} details` : "Community set details"}
            className={`absolute inset-0 flex min-h-0 flex-col transition-opacity duration-[440ms] ease-[cubic-bezier(0.22,1,0.36,1)] ${
              isDirectoryDetailOpen ? "opacity-100" : "opacity-0 pointer-events-none"
            }`}
          >
            {selectedDirectorySet ? (
              <div
                ref={detailContentScrollRef}
                className="balanced-scroll-gutter min-h-0 flex-1 overflow-y-auto pb-2"
                style={{ paddingTop: detailContentTopPadding }}
              >
                <div className="px-1 sm:px-2 md:px-3">
                  <section className="rounded-2xl px-4 py-4 sm:px-5 sm:py-5">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-white/65">Set Preview</p>
                      <div className="flex items-center gap-1.5">
                        <button
                          type="button"
                          onClick={goToPreviousDetailCarousel}
                          disabled={detailCarouselIndex === 0}
                          aria-label="Previous images"
                          className="grid h-8 w-8 place-items-center rounded-full bg-black/45 text-white/80 transition hover:bg-black/60 hover:text-white disabled:cursor-not-allowed disabled:opacity-35"
                        >
                          <i className="fa-solid fa-chevron-left text-[10px]" aria-hidden="true" />
                        </button>
                        <button
                          type="button"
                          onClick={goToNextDetailCarousel}
                          disabled={detailCarouselIndex >= maxDetailCarouselIndex}
                          aria-label="Next images"
                          className="grid h-8 w-8 place-items-center rounded-full bg-black/45 text-white/80 transition hover:bg-black/60 hover:text-white disabled:cursor-not-allowed disabled:opacity-35"
                        >
                          <i className="fa-solid fa-chevron-right text-[10px]" aria-hidden="true" />
                        </button>
                      </div>
                    </div>

                    <div className="mt-3 overflow-hidden rounded-xl">
                      <div
                        className="flex transition-transform duration-300 ease-out"
                        style={{
                          width: `${(detailCarouselImages.length / DETAIL_CAROUSEL_VISIBLE_COUNT) * 100}%`,
                          transform: `translateX(-${(detailCarouselIndex * 100) / detailCarouselImages.length}%)`,
                        }}
                      >
                        {detailCarouselImages.map((image, index) => (
                          <div
                            key={`${selectedDirectorySet.id}-detail-carousel-image-${index}`}
                            className="shrink-0 px-1 py-1"
                            style={{ width: `${100 / detailCarouselImages.length}%` }}
                          >
                            <img
                              src={image}
                              alt={`${selectedDirectorySet.title} preview ${index + 1}`}
                              className="h-[180px] w-full rounded-lg object-cover sm:h-[220px] md:h-[260px]"
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  </section>

                  <section className="mt-4 rounded-2xl px-4 py-4 sm:px-5 sm:py-5">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.1em] text-white/65">Information</p>
                    {selectedDirectorySet.tags.length > 0 ? (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {selectedDirectorySet.tags.map((tag) => (
                          <span key={`${selectedDirectorySet.id}-tag-${tag}`} className="rounded-full bg-white/10 px-2.5 py-1 text-[11px] text-white/72">
                            #{tag}
                          </span>
                        ))}
                      </div>
                    ) : null}
                    <div className="mt-3 space-y-3">
                      {DETAIL_LOREM_PARAGRAPHS.map((paragraph, index) => (
                        <p key={`${selectedDirectorySet.id}-detail-lorem-${index}`} className="text-sm leading-relaxed text-white/78">
                          {paragraph}
                        </p>
                      ))}
                    </div>
                  </section>
                </div>
              </div>
            ) : null}
          </section>
        </div>
      ) : (
        <>
          <div className="flex min-h-0 flex-1 flex-col pt-1 md:pt-2">
            <div className="shrink-0">
              <div className="flex flex-col gap-3 md:-mx-2 md:flex-row md:items-center md:justify-between md:gap-4 lg:-mx-3">
                <div className="w-full pl-5 sm:pl-6 md:w-auto md:pl-6 lg:pl-2">
                  <div className="flex items-center justify-center gap-2 md:justify-start">
                    <h2 className="text-xl font-semibold tracking-tight text-white sm:text-2xl md:text-[1.9rem]">Create Set</h2>
                    <span className="rounded-full border border-[#2b2b2b] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.09em] text-white/55">Beta</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-3 min-h-0 flex-1 overflow-hidden md:-mx-4 md:mt-4 lg:-mx-5">
              <div className="balanced-scroll-gutter h-full min-h-0 overflow-y-auto">
                <section className="rounded-3xl p-1 sm:p-2 md:p-3">
                  <div className="relative overflow-hidden rounded-2xl border border-[#2b2b2b] bg-transparent p-4 backdrop-blur-[2px] sm:p-5">
                  <div className="pointer-events-none absolute inset-0 bg-white/[0.04]" />
                  <div className="relative z-10">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <p className="text-[11px] font-semibold uppercase tracking-[0.11em] text-white/70">Create Set</p>
                        <p className="mt-1 text-sm text-white/64">Complete each step to publish your reel set.</p>
                      </div>
                      <span className="rounded-full border border-[#2b2b2b] bg-black/45 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.09em] text-white/72">
                        {completionPercent}% complete
                      </span>
                    </div>
                    <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                      <div className="h-full rounded-full bg-white transition-[width] duration-300" style={{ width: `${completionPercent}%` }} />
                    </div>
                    <div className="mt-3 grid gap-2 sm:grid-cols-2">
                      <span className={`rounded-lg border border-[#2b2b2b] px-2.5 py-2 text-[10px] font-semibold uppercase tracking-[0.08em] ${normalizedSetTitle ? "bg-[#74dfb4]/12 text-[#d4ffe9]" : "bg-black/35 text-white/62"}`}>
                        1. name your set
                      </span>
                      <span className={`rounded-lg border border-[#2b2b2b] px-2.5 py-2 text-[10px] font-semibold uppercase tracking-[0.08em] ${normalizedSetDescription.length >= MIN_SET_DESCRIPTION_LENGTH ? "bg-[#74dfb4]/12 text-[#d4ffe9]" : "bg-black/35 text-white/62"}`}>
                        2. add description
                      </span>
                      <span className={`rounded-lg border border-[#2b2b2b] px-2.5 py-2 text-[10px] font-semibold uppercase tracking-[0.08em] ${thumbnailPreview ? "bg-[#74dfb4]/12 text-[#d4ffe9]" : "bg-black/35 text-white/62"}`}>
                        3. upload thumbnail
                      </span>
                      <span className={`rounded-lg border border-[#2b2b2b] px-2.5 py-2 text-[10px] font-semibold uppercase tracking-[0.08em] ${validDraftReelCount > 0 && invalidDraftReelCount === 0 ? "bg-[#74dfb4]/12 text-[#d4ffe9]" : "bg-black/35 text-white/62"}`}>
                        4. add valid reels
                      </span>
                    </div>
                  </div>
                </div>

                <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,0.9fr)] lg:items-start">
                  <form onSubmit={onCreateSet} className="space-y-4 md:space-y-5">
                    <div className="relative overflow-hidden rounded-2xl border border-[#2b2b2b] bg-transparent p-4 backdrop-blur-[2px] sm:p-5">
                      <div className="pointer-events-none absolute inset-0 bg-white/[0.04]" />
                      <div className="relative z-10 space-y-5">
                      <label className="block">
                        <span className="mb-2 flex items-center justify-between gap-2 text-xs text-white/72">
                          <span>Set Name</span>
                          <span className="text-[10px] text-white/45">{normalizedSetTitle.length}/70</span>
                        </span>
                        <input
                          value={setTitle}
                          onChange={(event) => setSetTitle(event.target.value)}
                          maxLength={70}
                          placeholder="Example: Organic Chemistry Reactions"
                          className="h-11 w-full rounded-xl border border-[#2b2b2b] bg-black/55 px-3 text-sm text-white outline-none placeholder:text-white/40 transition-colors focus:border-[#2b2b2b]"
                        />
                      </label>

                      <label className="block">
                        <span className="mb-2 flex items-center justify-between gap-2 text-xs text-white/72">
                          <span>Description</span>
                          <span className={`text-[10px] ${normalizedSetDescription.length >= MIN_SET_DESCRIPTION_LENGTH ? "text-[#9ef8cb]" : "text-white/45"}`}>
                            {normalizedSetDescription.length} / {MIN_SET_DESCRIPTION_LENGTH} min
                          </span>
                        </span>
                        <textarea
                          value={setDescription}
                          onChange={(event) => setSetDescription(event.target.value)}
                          placeholder="What does this set cover and who is it for?"
                          className="h-24 w-full resize-none rounded-xl border border-[#2b2b2b] bg-black/55 px-3 py-2 text-sm text-white outline-none placeholder:text-white/40 transition-colors focus:border-[#2b2b2b] md:h-24"
                        />
                        {normalizedSetDescription.length === 0 ? (
                          <p className="mt-1.5 text-[11px] text-zinc-400">
                            Description must be at least {MIN_SET_DESCRIPTION_LENGTH} characters.
                          </p>
                        ) : descriptionHasTooFewChars ? (
                          <p className="mt-1.5 text-[11px] text-zinc-400">
                            Description must be at least {MIN_SET_DESCRIPTION_LENGTH} characters. Add {descriptionCharsRemaining} more
                            {descriptionCharsRemaining === 1 ? " character." : " characters."}
                          </p>
                        ) : null}
                      </label>

                      <label className="block">
                        <span className="mb-2 block text-xs text-white/72">Tags</span>
                        <input
                          value={setTags}
                          onChange={(event) => setSetTags(event.target.value)}
                          placeholder="chemistry, reaction mechanisms, exam prep"
                          className="h-11 w-full rounded-xl border border-[#2b2b2b] bg-black/55 px-3 text-sm text-white outline-none placeholder:text-white/40 transition-colors focus:border-[#2b2b2b]"
                        />
                        <p className="mt-1.5 text-[11px] text-zinc-400">Add commas to add new tags.</p>
                        {parsedSetTags.length > 0 ? (
                          <div className="mt-2 flex flex-wrap gap-1.5">
                            {parsedSetTags.map((tag) => (
                              <span key={`create-tag-${tag}`} className="rounded-full border border-[#2b2b2b] bg-white/6 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/75">
                                #{tag}
                              </span>
                            ))}
                          </div>
                        ) : null}
                      </label>

                      <div>
                        <p className="mb-2 block text-xs text-white/72">Thumbnail Image</p>
                        <input id="community-set-thumbnail" type="file" accept="image/*" className="sr-only" onChange={onThumbnailFileChange} />
                        <label
                          htmlFor="community-set-thumbnail"
                          onDragEnter={onThumbnailDragEnter}
                          onDragOver={onThumbnailDragOver}
                          onDragLeave={onThumbnailDragLeave}
                          onDrop={onThumbnailDrop}
                          className={`group relative block h-[220px] w-full cursor-pointer overflow-hidden rounded-xl border border-dashed ${
                            isThumbnailDragOver ? "border-white/60 bg-white/10" : "border-[#2b2b2b] bg-black/55"
                          } sm:h-[250px]`}
                        >
                          {thumbnailPreview ? (
                            <img
                              src={thumbnailPreview}
                              alt="Set thumbnail preview"
                              className="h-full w-full object-cover transition opacity-100"
                            />
                          ) : (
                            <div className="grid h-full w-full place-items-center bg-[linear-gradient(145deg,rgba(255,255,255,0.16),rgba(255,255,255,0.04))] text-white/70 transition group-hover:text-white/85">
                              <i className="fa-regular fa-image -translate-y-6 text-lg sm:-translate-y-7" aria-hidden="true" />
                            </div>
                          )}
                          <span className="absolute inset-0 grid place-items-center bg-black/35 text-white/85">
                            <span className="flex max-w-[90%] translate-y-5 flex-col items-center text-center sm:translate-y-6">
                              <span className={`truncate text-sm font-semibold ${thumbnailPreview ? "text-white" : "text-white/85"}`}>
                                {thumbnailPreview ? thumbnailFileName || "Image selected" : "Drag and drop your image here"}
                              </span>
                              <span className="mt-1 text-xs text-white/58">
                                {thumbnailPreview ? "Click to replace image" : "Or click to browse (PNG, JPG, WEBP)"}
                              </span>
                            </span>
                          </span>
                        </label>
                        <p className="mt-2 text-[11px] text-zinc-400">Use a vertical image for better mobile previews.</p>
                        {thumbnailPreview ? (
                          <button
                            type="button"
                            onClick={() => {
                              setThumbnailPreview("");
                              setThumbnailFileName("");
                            }}
                            className="mt-2 inline-flex items-center gap-1 rounded-lg border border-[#2b2b2b] bg-black/45 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/72 transition hover:bg-white/10 hover:text-white"
                          >
                            <i className="fa-solid fa-trash text-[9px]" aria-hidden="true" />
                            Remove
                          </button>
                        ) : null}
                      </div>
                    </div>
                    </div>

                    <div className="relative min-h-0 overflow-hidden rounded-2xl border border-[#2b2b2b] bg-transparent p-3.5 backdrop-blur-[2px] sm:p-4">
                      <div className="pointer-events-none absolute inset-0 bg-white/[0.04]" />
                      <div className="relative z-10">
                      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                        <p className="text-[10px] font-semibold uppercase tracking-[0.09em] text-white/68">Embed Reels ({SUPPORTED_PLATFORMS_LABEL})</p>
                        <p className="text-[10px] font-semibold uppercase tracking-[0.08em] text-white/55">
                          {validDraftReelCount} valid / {invalidDraftReelCount} invalid
                        </p>
                      </div>
                      <div className="balanced-scroll-gutter max-h-[320px] space-y-3 overflow-y-auto">
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
                                  className="h-10 w-full rounded-lg bg-black/60 px-2.5 text-xs text-white outline-none placeholder:text-white/40 sm:h-9"
                                />
                                <button
                                  type="button"
                                  onClick={() => removeReelInputRow(row.id)}
                                  className="inline-flex h-9 w-full shrink-0 items-center justify-center gap-1 rounded-lg bg-black/55 text-white/72 transition hover:bg-white/10 hover:text-white sm:grid sm:h-8 sm:w-8 sm:place-items-center"
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
                      <button
                        type="button"
                        onClick={addReelInputRow}
                        className="mt-3 inline-flex items-center gap-1 px-1 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/72 transition hover:text-white"
                      >
                        <i className="fa-solid fa-plus text-[9px]" aria-hidden="true" />
                        Add Reels
                      </button>
                    </div>
                    </div>

                    {createError ? <p className="text-xs text-[#ffb4b4]">{createError}</p> : null}
                    {createSuccess ? <p className="text-xs text-[#9ef8cb]">{createSuccess}</p> : null}

                    <button
                      type="submit"
                      disabled={!canPostSet}
                      className={`inline-flex h-11 w-full items-center justify-center rounded-xl border px-4 text-sm font-semibold transition ${
                        canPostSet
                          ? "border-[#2b2b2b] bg-black/55 text-white hover:bg-white hover:text-black"
                          : "cursor-not-allowed border-[#2b2b2b] bg-black/35 text-white/45"
                      }`}
                    >
                      {isPostingSet ? "Posting..." : "Post Community Set"}
                    </button>
                  </form>

                  <aside className="relative overflow-hidden rounded-2xl border border-[#2b2b2b] bg-transparent p-4 backdrop-blur-[2px] sm:p-5 lg:sticky lg:top-3">
                    <div className="pointer-events-none absolute inset-0 bg-white/[0.04]" />
                    <div className="relative z-10">
                      <p className="text-[10px] font-semibold uppercase tracking-[0.1em] text-white/62">Live Preview</p>
                      <div className="mt-3 overflow-hidden rounded-xl border border-[#2b2b2b] bg-black/45">
                        {thumbnailPreview ? (
                          <img
                            src={thumbnailPreview}
                            alt="Draft set cover"
                            className="h-[220px] w-full object-cover"
                          />
                        ) : (
                          <div className="grid h-[220px] w-full place-items-center bg-[linear-gradient(145deg,rgba(255,255,255,0.16),rgba(255,255,255,0.04))] text-white/70">
                            <i className="fa-regular fa-image text-lg" aria-hidden="true" />
                          </div>
                        )}
                      </div>
                      <h3 className="mt-3 text-lg font-semibold leading-tight text-white">
                        {normalizedSetTitle || "Your set title"}
                      </h3>
                      <p className="mt-2 text-sm leading-relaxed text-white/70">
                        {normalizedSetDescription || "Add a description to show what learners will get from this set."}
                      </p>
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {parsedSetTags.length > 0
                          ? parsedSetTags.map((tag) => (
                            <span key={`preview-tag-${tag}`} className="rounded-full bg-white/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-white/72">
                              #{tag}
                            </span>
                          ))
                          : null}
                      </div>
                      <div className="mt-4 grid grid-cols-2 gap-2">
                        <div className="rounded-lg border border-[#2b2b2b] bg-black/35 px-2.5 py-2">
                          <p className="text-[10px] uppercase tracking-[0.08em] text-white/52">Reels</p>
                          <p className="mt-1 text-sm font-semibold text-white">{validDraftReelCount}</p>
                        </div>
                        <div className="rounded-lg border border-[#2b2b2b] bg-black/35 px-2.5 py-2">
                          <p className="text-[10px] uppercase tracking-[0.08em] text-white/52">Status</p>
                          <p className={`mt-1 text-sm font-semibold ${canPostSet ? "text-[#9ef8cb]" : "text-white/76"}`}>
                            {canPostSet ? "Ready to post" : "Draft"}
                          </p>
                        </div>
                      </div>
                    </div>
                  </aside>
                </div>
              </section>
            </div>
          </div>
          </div>
      </>
    )}
      {detailBannerPortal}
    </div>
  );
}
