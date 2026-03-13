"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { CommunityAccountScreen } from "@/components/CommunityAccountScreen";
import { FullscreenLoadingScreen } from "@/components/FullscreenLoadingScreen";
import { COMMUNITY_AUTH_CHANGED_EVENT, fetchCommunityAccount, readCommunityAuthSession } from "@/lib/api";
import { useLoadingScreenGate } from "@/lib/useLoadingScreenGate";
import type { CommunityAccount } from "@/lib/types";

function normalizeReturnTab(value: string | null): "search" | "community" | "edit" | "settings" | null {
  if (value === "search" || value === "community" || value === "edit" || value === "settings") {
    return value;
  }
  return null;
}

function AccountPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const returnTab = useMemo(() => normalizeReturnTab(searchParams.get("return_tab")), [searchParams]);
  const [communityAccount, setCommunityAccount] = useState<CommunityAccount | null>(null);
  const [accountScreenReady, setAccountScreenReady] = useState(false);
  const showLoadingScreen = useLoadingScreenGate(accountScreenReady, { minimumVisibleMs: 1000 });

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    let cancelled = false;
    const finishBootstrap = () => {
      if (!cancelled) {
        setAccountScreenReady(true);
      }
    };
    const syncAccountFromStorage = () => {
      const stored = readCommunityAuthSession();
      if (!cancelled) {
        setCommunityAccount(stored?.account ?? null);
      }
    };
    const validateStoredAccount = async () => {
      const stored = readCommunityAuthSession();
      if (!stored?.sessionToken) {
        if (!cancelled) {
          setCommunityAccount(null);
        }
        finishBootstrap();
        return;
      }
      try {
        const account = await fetchCommunityAccount();
        if (!cancelled) {
          setCommunityAccount(account);
        }
      } catch {
        syncAccountFromStorage();
      } finally {
        finishBootstrap();
      }
    };
    const onStorage = (event: StorageEvent) => {
      if (event.storageArea !== window.localStorage) {
        return;
      }
      syncAccountFromStorage();
    };
    const onAuthChanged = () => {
      syncAccountFromStorage();
    };
    syncAccountFromStorage();
    void validateStoredAccount();
    window.addEventListener("storage", onStorage);
    window.addEventListener(COMMUNITY_AUTH_CHANGED_EVENT, onAuthChanged);
    return () => {
      cancelled = true;
      window.removeEventListener("storage", onStorage);
      window.removeEventListener(COMMUNITY_AUTH_CHANGED_EVENT, onAuthChanged);
    };
  }, []);

  const backTarget = returnTab ? `/?tab=${returnTab}` : "/";
  const onBack = useCallback(() => {
    router.push(backTarget);
  }, [backTarget, router]);
  const onOpenYourSets = useCallback(() => {
    router.push("/?tab=edit");
  }, [router]);

  if (showLoadingScreen) {
    return <FullscreenLoadingScreen />;
  }

  return (
    <CommunityAccountScreen
      account={communityAccount}
      onBack={onBack}
      onAccountChange={setCommunityAccount}
      onOpenYourSets={onOpenYourSets}
    />
  );
}

export default function AccountPage() {
  return (
    <Suspense fallback={<FullscreenLoadingScreen />}>
      <AccountPageContent />
    </Suspense>
  );
}
