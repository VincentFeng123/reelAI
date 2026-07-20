"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { CommunityAccountScreen } from "@/components/CommunityAccountScreen";
import { FullscreenLoadingScreen } from "@/components/FullscreenLoadingScreen";
import { COMMUNITY_AUTH_CHANGED_EVENT, readCommunityAuthSession, restoreCommunityAccountFromSessionToken } from "@/lib/api";
import { useLoadingScreenGate } from "@/lib/useLoadingScreenGate";
import type { CommunityAccount } from "@/lib/types";

function normalizeReturnTab(value: string | null): "search" | "community" | "edit" | "settings" | null {
  if (value === "search" || value === "community" || value === "edit" || value === "settings") {
    return value;
  }
  return null;
}

function normalizeAccountView(value: string | null): "default" | "change-password" | "delete-account" | "billing" {
  if (value === "change-password" || value === "delete-account" || value === "billing") {
    return value;
  }
  return "default";
}

function AccountPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const returnTab = useMemo(() => normalizeReturnTab(searchParams.get("return_tab")), [searchParams]);
  const requestedView = useMemo(() => normalizeAccountView(searchParams.get("view")), [searchParams]);
  const requestedAuthMode = searchParams.get("mode") === "register" ? "register" : "login";
  const [communityAccount, setCommunityAccount] = useState<CommunityAccount | null>(null);
  const [accountScreenReady, setAccountScreenReady] = useState(false);
  const [authCompleted, setAuthCompleted] = useState(false);
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
      try {
        // Always probe the current tab's session token. Unverified logins keep
        // their token without persisting account metadata, so /auth/me is the
        // only authoritative way to restore their verification flow on reload.
        const account = await restoreCommunityAccountFromSessionToken();
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

  const postAuthTarget = useMemo(() => {
    if (returnTab === "settings") {
      return "/?settings=account";
    }
    if (returnTab) {
      return `/?tab=${returnTab}`;
    }
    return "/";
  }, [returnTab]);

  useEffect(() => {
    if (requestedView === "billing") {
      router.replace("/?settings=plan");
      return;
    }
    if (communityAccount?.isVerified) {
      router.replace(authCompleted ? postAuthTarget : "/?settings=account");
    }
  }, [authCompleted, communityAccount?.isVerified, postAuthTarget, requestedView, router]);

  const accountBaseTarget = returnTab ? `/account?return_tab=${returnTab}` : "/account";
  const backTarget = requestedView === "change-password" || requestedView === "delete-account"
    ? accountBaseTarget
    : returnTab === "settings"
      ? "/?settings=account"
      : returnTab
      ? `/?tab=${returnTab}`
      : "/";
  const changePasswordTarget = returnTab
    ? `/account?view=change-password&return_tab=${returnTab}`
    : "/account?view=change-password";
  const deleteAccountTarget = returnTab
    ? `/account?view=delete-account&return_tab=${returnTab}`
    : "/account?view=delete-account";
  const accountView = communityAccount?.isVerified && requestedView !== "billing" ? requestedView : "default";
  const onBack = useCallback(() => {
    router.push(backTarget);
  }, [backTarget, router]);
  const onOpenChangePassword = useCallback(() => {
    router.push(changePasswordTarget);
  }, [changePasswordTarget, router]);
  const onOpenDeleteAccount = useCallback(() => {
    router.push(deleteAccountTarget);
  }, [deleteAccountTarget, router]);
  const onAccountChange = useCallback((nextAccount: CommunityAccount | null) => {
    setCommunityAccount(nextAccount);
    if (nextAccount?.isVerified) {
      setAuthCompleted(true);
    }
  }, []);
  const onAuthModeChange = useCallback((mode: "login" | "register") => {
    const nextQuery = new URLSearchParams();
    if (mode === "register") {
      nextQuery.set("mode", "register");
    }
    if (returnTab) {
      nextQuery.set("return_tab", returnTab);
    }
    const query = nextQuery.toString();
    router.replace(query ? `/account?${query}` : "/account");
  }, [returnTab, router]);

  if (showLoadingScreen || requestedView === "billing" || communityAccount?.isVerified) {
    return <FullscreenLoadingScreen />;
  }

  return (
    <CommunityAccountScreen
      account={communityAccount}
      view={accountView}
      initialAuthMode={requestedAuthMode}
      onBack={onBack}
      onAccountChange={onAccountChange}
      onAuthModeChange={onAuthModeChange}
      onOpenChangePassword={onOpenChangePassword}
      onOpenDeleteAccount={onOpenDeleteAccount}
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
