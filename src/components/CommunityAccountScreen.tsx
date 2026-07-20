"use client";

import { type ChangeEvent, type ClipboardEvent as ReactClipboardEvent, type FormEvent, type KeyboardEvent as ReactKeyboardEvent, useCallback, useEffect, useRef, useState } from "react";

import {
  changeCommunityVerificationEmail,
  changeCommunityPassword,
  deleteCommunityAccount,
  loginCommunityAccount,
  logoutCommunityAccount,
  registerCommunityAccount,
  resendCommunityVerification,
  sendCommunitySignupVerification,
  verifyCommunityAccount,
  verifyCommunitySignupEmail,
} from "@/lib/api";
import { BillingPlanUsageCard } from "@/components/BillingPlanUsageCard";
import { FadePresence } from "@/components/FadePresence";
import { ViewportModalPortal } from "@/components/ViewportModalPortal";
import type { CommunityAccount } from "@/lib/types";

type CommunityAccountScreenProps = {
  account: CommunityAccount | null;
  view: "default" | "change-password" | "delete-account";
  initialAuthMode?: "login" | "register";
  onBack: () => void;
  onAccountChange: (account: CommunityAccount | null) => void;
  onAuthModeChange?: (mode: "login" | "register") => void;
  onOpenChangePassword: () => void;
  onOpenDeleteAccount: () => void;
};

const AUTH_MODE_FADE_MS = 420;
const VERIFICATION_CODE_LENGTH = 6;

function getAuthModeFadeMs(): number {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : AUTH_MODE_FADE_MS;
}

function normalizeSignupEmailForComparison(value: string): string {
  return value.trim().toLowerCase();
}

export function CommunityAccountScreen({
  account,
  view,
  initialAuthMode = "login",
  onBack,
  onAccountChange,
  onAuthModeChange,
  onOpenChangePassword,
  onOpenDeleteAccount,
}: CommunityAccountScreenProps) {
  const [authMode, setAuthMode] = useState<"register" | "login">(initialAuthMode);
  const [displayView, setDisplayView] = useState<"default" | "change-password" | "delete-account">(view);
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [verificationCode, setVerificationCode] = useState("");
  const [verificationCodeDebug, setVerificationCodeDebug] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [sendVerificationBusy, setSendVerificationBusy] = useState(false);
  const [verificationBusy, setVerificationBusy] = useState(false);
  const [resendBusy, setResendBusy] = useState(false);
  const [changeEmailBusy, setChangeEmailBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [isVerificationModalOpen, setIsVerificationModalOpen] = useState(false);
  const [verificationFlow, setVerificationFlow] = useState<"account" | "signup" | null>(null);
  const [pendingVerificationEmail, setPendingVerificationEmail] = useState<string | null>(null);
  const [verifiedSignupEmail, setVerifiedSignupEmail] = useState<string | null>(null);
  const [authContentVisible, setAuthContentVisible] = useState(true);
  const [authModeTransitioning, setAuthModeTransitioning] = useState(false);
  const authModeTransitionTimerRef = useRef<number | null>(null);
  const authModeTransitionFrameRef = useRef<number | null>(null);
  const pendingAuthModeFocusRef = useRef<"register" | "login" | null>(null);
  const authModeFirstFieldRefs = useRef<Record<"register" | "login", HTMLInputElement | null>>({
    login: null,
    register: null,
  });
  const [showChangeEmail, setShowChangeEmail] = useState(false);
  const [changeEmailValue, setChangeEmailValue] = useState("");
  const [changeEmailPassword, setChangeEmailPassword] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [changePasswordBusy, setChangePasswordBusy] = useState(false);
  const [changePasswordError, setChangePasswordError] = useState<string | null>(null);
  const [changePasswordSuccess, setChangePasswordSuccess] = useState<string | null>(null);
  const [deleteAccountPassword, setDeleteAccountPassword] = useState("");
  const [deleteAccountBusy, setDeleteAccountBusy] = useState(false);
  const [deleteAccountError, setDeleteAccountError] = useState<string | null>(null);
  const authSubmitInFlightRef = useRef(false);
  const verificationDigitInputRefs = useRef<Array<HTMLInputElement | null>>([]);
  const verificationDialogRef = useRef<HTMLDivElement | null>(null);
  const verificationReturnFocusRef = useRef<HTMLElement | null>(null);

  const openVerificationModal = useCallback(() => {
    verificationReturnFocusRef.current = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    setIsVerificationModalOpen(true);
  }, []);

  const restoreVerificationFocus = useCallback(() => {
    const returnFocus = verificationReturnFocusRef.current;
    verificationReturnFocusRef.current = null;
    window.requestAnimationFrame(() => {
      if (returnFocus?.isConnected) {
        returnFocus.focus();
      }
    });
  }, []);

  const closeVerificationModal = useCallback(() => {
    if (verificationBusy) {
      return;
    }
    setIsVerificationModalOpen(false);
    if (!account) {
      setVerificationFlow(null);
    }
    restoreVerificationFocus();
  }, [account, restoreVerificationFocus, verificationBusy]);

  useEffect(() => {
    setAuthMode(initialAuthMode);
  }, [initialAuthMode]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Tab" && isVerificationModalOpen) {
        const dialog = verificationDialogRef.current;
        if (!dialog) {
          return;
        }
        const focusable = Array.from(dialog.querySelectorAll<HTMLElement>(
          "a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])",
        ));
        if (focusable.length === 0) {
          event.preventDefault();
          dialog.focus();
          return;
        }
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        const activeElement = document.activeElement;
        if (event.shiftKey && (activeElement === first || !dialog.contains(activeElement))) {
          event.preventDefault();
          last.focus();
          return;
        }
        if (!event.shiftKey && (activeElement === last || !dialog.contains(activeElement))) {
          event.preventDefault();
          first.focus();
        }
        return;
      }
      if (event.key === "Escape" && isVerificationModalOpen) {
        event.preventDefault();
        event.stopPropagation();
        if (!verificationBusy) {
          closeVerificationModal();
        }
        return;
      }
      if (event.key === "Escape" && !busy && !sendVerificationBusy && !verificationBusy && !resendBusy && !changeEmailBusy && !deleteAccountBusy) {
        onBack();
      }
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
    };
  }, [busy, changeEmailBusy, closeVerificationModal, deleteAccountBusy, isVerificationModalOpen, onBack, resendBusy, sendVerificationBusy, verificationBusy]);

  useEffect(() => {
    setPassword("");
    setError(null);
    setSuccess(null);
    setVerificationCode("");
    setSendVerificationBusy(false);
    setPendingVerificationEmail(null);
    setVerificationFlow(null);
    setVerifiedSignupEmail(null);
    setShowChangeEmail(false);
    setChangeEmailValue(account?.email ?? "");
    setChangeEmailPassword("");
    setCurrentPassword("");
    setNewPassword("");
    setChangePasswordError(null);
    setChangePasswordSuccess(null);
    setDeleteAccountPassword("");
    setDeleteAccountError(null);
    if (account) {
      setEmail(account.email ?? "");
      if (account.isVerified) {
        setVerificationCodeDebug(null);
      }
    } else {
      setEmail("");
      setUsername("");
      setVerificationCodeDebug(null);
    }
    authSubmitInFlightRef.current = false;
  }, [account, view]);

  useEffect(() => {
    return () => {
      if (authModeTransitionTimerRef.current !== null) {
        window.clearTimeout(authModeTransitionTimerRef.current);
      }
      if (authModeTransitionFrameRef.current !== null) {
        window.cancelAnimationFrame(authModeTransitionFrameRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (authModeTransitioning || pendingAuthModeFocusRef.current !== authMode) {
      return;
    }
    pendingAuthModeFocusRef.current = null;
    authModeFirstFieldRefs.current[authMode]?.focus();
  }, [authMode, authModeTransitioning]);

  useEffect(() => {
    if (verificationFlow === "account" && (!account || account.isVerified)) {
      setIsVerificationModalOpen(false);
      setVerificationFlow(null);
    }
  }, [account, verificationFlow]);

  useEffect(() => {
    if (!verifiedSignupEmail) {
      return;
    }
    if (normalizeSignupEmailForComparison(email) !== verifiedSignupEmail) {
      setVerifiedSignupEmail(null);
    }
  }, [email, verifiedSignupEmail]);

  useEffect(() => {
    if (!isVerificationModalOpen) {
      return;
    }
    const focusIndex = Math.min(verificationCode.trim().length, VERIFICATION_CODE_LENGTH - 1);
    const target = verificationDigitInputRefs.current[focusIndex];
    if (!target) {
      return;
    }
    const frameId = window.requestAnimationFrame(() => {
      target.focus();
      target.select();
    });
    return () => {
      window.cancelAnimationFrame(frameId);
    };
  }, [isVerificationModalOpen, verificationCode]);

  useEffect(() => {
    if (view === displayView) {
      return;
    }
    if (authModeTransitionTimerRef.current !== null) {
      window.clearTimeout(authModeTransitionTimerRef.current);
    }
    if (authModeTransitionFrameRef.current !== null) {
      window.cancelAnimationFrame(authModeTransitionFrameRef.current);
    }
    setAuthModeTransitioning(true);
    setAuthContentVisible(false);
    const fadeMs = getAuthModeFadeMs();
    authModeTransitionTimerRef.current = window.setTimeout(() => {
      setDisplayView(view);
      const revealContent = () => {
        setAuthContentVisible(true);
        authModeTransitionFrameRef.current = null;
        if (fadeMs === 0) {
          setAuthModeTransitioning(false);
          authModeTransitionTimerRef.current = null;
          return;
        }
        authModeTransitionTimerRef.current = window.setTimeout(() => {
          setAuthModeTransitioning(false);
          authModeTransitionTimerRef.current = null;
        }, fadeMs);
      };
      if (fadeMs === 0) {
        revealContent();
        return;
      }
      authModeTransitionFrameRef.current = window.requestAnimationFrame(() => {
        authModeTransitionFrameRef.current = window.requestAnimationFrame(revealContent);
      });
    }, fadeMs);
  }, [displayView, view]);

  const submitCommunityAuth = useCallback(async () => {
    if (busy || sendVerificationBusy || authSubmitInFlightRef.current) {
      return;
    }
    const trimmedUsername = username.trim();
    const trimmedEmail = email.trim();
    if (!trimmedUsername || !password || (authMode === "register" && !trimmedEmail)) {
      setError(authMode === "register" ? "Enter a username, email, and password." : "Enter a username and password.");
      setSuccess(null);
      return;
    }
    if (authMode === "register" && normalizeSignupEmailForComparison(trimmedEmail) !== verifiedSignupEmail) {
      setError("Verify your email before creating the account.");
      setSuccess(null);
      return;
    }
    authSubmitInFlightRef.current = true;
    setBusy(true);
    setError(null);
    setSuccess(null);
    try {
      const session = authMode === "register"
        ? await registerCommunityAccount({
          username: trimmedUsername,
          email: trimmedEmail,
          password,
        })
        : await loginCommunityAccount({
          username: trimmedUsername,
          password,
          persistAccountWhenUnverified: false,
        });
      setPassword("");
      setVerificationCode("");
      setVerificationCodeDebug(session.verificationCodeDebug ?? null);
      if (session.verificationRequired || !session.account.isVerified) {
        const targetEmail = session.account.email ?? (authMode === "register" ? trimmedEmail : "");
        setPendingVerificationEmail(targetEmail || null);
        setVerificationFlow("account");
        openVerificationModal();
        setSuccess(
          targetEmail
            ? null
            : "Verify your email to unlock Your Sets.",
        );
      } else if (session.claimedLegacySets > 0) {
        onAccountChange(session.account);
        setVerifiedSignupEmail(null);
        setSuccess(
          `${authMode === "register" ? "Account created" : "Signed in"} and linked ${session.claimedLegacySets} existing set${session.claimedLegacySets === 1 ? "" : "s"} from this device.`,
        );
      } else {
        onAccountChange(session.account);
        setVerifiedSignupEmail(null);
        setSuccess(authMode === "register" ? "Account created." : "Signed in.");
      }
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Could not complete that request.");
      setSuccess(null);
    } finally {
      authSubmitInFlightRef.current = false;
      setBusy(false);
    }
  }, [authMode, busy, email, onAccountChange, openVerificationModal, password, sendVerificationBusy, username, verifiedSignupEmail]);

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    await submitCommunityAuth();
  };

  const onSendVerificationEmail = async () => {
    if (busy || sendVerificationBusy) {
      return;
    }
    const trimmedEmail = email.trim();
    if (!trimmedEmail) {
      setError("Enter your email address.");
      setSuccess(null);
      return;
    }
    setSendVerificationBusy(true);
    setError(null);
    setSuccess(null);
    try {
      const result = await sendCommunitySignupVerification({
        email: trimmedEmail,
        username: username.trim() || undefined,
      });
      setVerificationCode("");
      setVerificationCodeDebug(result.verificationCodeDebug);
      setPendingVerificationEmail(result.email);
      if (result.verified || !result.verificationRequired) {
        setVerifiedSignupEmail(normalizeSignupEmailForComparison(result.email));
        setVerificationFlow(null);
        setIsVerificationModalOpen(false);
        setSuccess("Email verified. Click Create Account to continue.");
      } else {
        setVerifiedSignupEmail(null);
        setVerificationFlow("signup");
        openVerificationModal();
        setSuccess(null);
      }
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Could not send a verification code.");
      setSuccess(null);
    } finally {
      setSendVerificationBusy(false);
    }
  };

  const onSignupEmailChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const nextEmail = event.target.value;
    const nextNormalizedEmail = normalizeSignupEmailForComparison(nextEmail);
    setEmail(nextEmail);
    if (
      verifiedSignupEmail === null
      && pendingVerificationEmail === null
      && verificationFlow !== "signup"
      && verificationCodeDebug === null
    ) {
      return;
    }
    if (
      verifiedSignupEmail !== nextNormalizedEmail
      || pendingVerificationEmail !== null
      || verificationFlow === "signup"
      || verificationCodeDebug !== null
    ) {
      setVerifiedSignupEmail(null);
      setPendingVerificationEmail(null);
      setVerificationCode("");
      setVerificationCodeDebug(null);
      setSuccess(null);
      if (verificationFlow === "signup") {
        setVerificationFlow(null);
        setIsVerificationModalOpen(false);
      }
    }
  }, [pendingVerificationEmail, verificationCodeDebug, verificationFlow, verifiedSignupEmail]);

  const onSignOut = async () => {
    if (busy || verificationBusy || resendBusy || changeEmailBusy || changePasswordBusy || deleteAccountBusy) {
      return;
    }
    setBusy(true);
    setError(null);
    setSuccess(null);
    setVerificationCodeDebug(null);
    setShowChangeEmail(false);
    setChangeEmailPassword("");
    setDeleteAccountPassword("");
    setDeleteAccountError(null);
    try {
      await logoutCommunityAccount();
      onAccountChange(null);
    } finally {
      setBusy(false);
    }
  };

  const onChangePassword = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (changePasswordBusy) {
      return;
    }
    if (!currentPassword || !newPassword) {
      setChangePasswordError("Enter your current and new password.");
      setChangePasswordSuccess(null);
      return;
    }
    if (newPassword.length < 8) {
      setChangePasswordError("New password must be at least 8 characters.");
      setChangePasswordSuccess(null);
      return;
    }
    setChangePasswordBusy(true);
    setChangePasswordError(null);
    setChangePasswordSuccess(null);
    try {
      await changeCommunityPassword({ currentPassword, newPassword });
      setCurrentPassword("");
      setNewPassword("");
      setChangePasswordSuccess("Password changed. Other devices have been signed out.");
    } catch (submitError) {
      setChangePasswordError(submitError instanceof Error ? submitError.message : "Could not change password.");
      setChangePasswordSuccess(null);
    } finally {
      setChangePasswordBusy(false);
    }
  };

  const onDeleteAccount = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (deleteAccountBusy) {
      return;
    }
    if (!deleteAccountPassword) {
      setDeleteAccountError("Enter your current password to delete this account.");
      return;
    }
    setDeleteAccountBusy(true);
    setDeleteAccountError(null);
    setError(null);
    setSuccess(null);
    try {
      await deleteCommunityAccount({ currentPassword: deleteAccountPassword });
      onAccountChange(null);
    } catch (submitError) {
      setDeleteAccountError(submitError instanceof Error ? submitError.message : "Could not delete this account.");
    } finally {
      setDeleteAccountBusy(false);
    }
  };

  const onVerifyAccount = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (verificationBusy) {
      return;
    }
    const trimmedCode = verificationCode.trim();
    if (!trimmedCode) {
      setError("Enter the verification code.");
      setSuccess(null);
      return;
    }
    setVerificationBusy(true);
    setError(null);
    setSuccess(null);
    try {
      if (verificationFlow === "signup") {
        const targetEmail = (pendingVerificationEmail || email).trim();
        if (!targetEmail) {
          throw new Error("Enter your email address before verifying.");
        }
        const result = await verifyCommunitySignupEmail({
          email: targetEmail,
          code: trimmedCode,
        });
        if (!result.verified) {
          throw new Error("Verification code is incorrect.");
        }
        setVerifiedSignupEmail(normalizeSignupEmailForComparison(result.email));
        setVerificationCode("");
        setVerificationCodeDebug(null);
        setPendingVerificationEmail(result.email);
        setVerificationFlow(null);
        setIsVerificationModalOpen(false);
        restoreVerificationFocus();
        setSuccess("Email verified. Click Create Account to continue.");
        return;
      }
      const result = await verifyCommunityAccount({ code: trimmedCode });
      onAccountChange(result.account);
      setVerificationCode("");
      setVerificationCodeDebug(null);
      setPendingVerificationEmail(null);
      setVerificationFlow(null);
      setIsVerificationModalOpen(false);
      restoreVerificationFocus();
      if (result.claimedLegacySets > 0) {
        setSuccess(`Account verified and linked ${result.claimedLegacySets} existing set${result.claimedLegacySets === 1 ? "" : "s"} from this device.`);
      } else {
        setSuccess("Account verified. Your Sets are now unlocked.");
      }
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Could not verify this account.");
      setSuccess(null);
    } finally {
      setVerificationBusy(false);
    }
  };

  const onResendVerification = async () => {
    if (resendBusy) {
      return;
    }
    setResendBusy(true);
    setError(null);
    setSuccess(null);
    try {
      const result = await resendCommunityVerification();
      onAccountChange(result.account);
      setVerificationCodeDebug(result.verificationCodeDebug);
      setVerificationCode("");
      setPendingVerificationEmail(result.account.email ?? null);
      setVerificationFlow("account");
      openVerificationModal();
      setSuccess(
        result.account.email
          ? `Sent a new verification code to ${result.account.email}.`
          : "Sent a new verification code.",
      );
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Could not resend verification code.");
      setSuccess(null);
    } finally {
      setResendBusy(false);
    }
  };

  const onChangeVerificationEmail = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (changeEmailBusy) {
      return;
    }
    const trimmedEmail = changeEmailValue.trim();
    if (!trimmedEmail || !changeEmailPassword) {
      setError("Enter your new email and current password.");
      setSuccess(null);
      return;
    }
    setChangeEmailBusy(true);
    setError(null);
    setSuccess(null);
    try {
      const result = await changeCommunityVerificationEmail({
        email: trimmedEmail,
        currentPassword: changeEmailPassword,
      });
      onAccountChange(result.account);
      setChangeEmailPassword("");
      setVerificationCode("");
      setVerificationCodeDebug(result.verificationCodeDebug);
      setPendingVerificationEmail(result.account.email ?? null);
      setShowChangeEmail(false);
      setVerificationFlow("account");
      openVerificationModal();
      setSuccess(
        result.account.email
          ? `Verification email updated. Enter the code sent to ${result.account.email}.`
          : "Verification email updated.",
      );
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Could not update verification email.");
      setSuccess(null);
    } finally {
      setChangeEmailBusy(false);
    }
  };

  const switchAuthMode = useCallback((nextMode: "register" | "login") => {
    if (nextMode === authMode || authModeTransitioning) {
      return;
    }
    if (authModeTransitionTimerRef.current !== null) {
      window.clearTimeout(authModeTransitionTimerRef.current);
    }
    if (authModeTransitionFrameRef.current !== null) {
      window.cancelAnimationFrame(authModeTransitionFrameRef.current);
    }
    setPassword("");
    setError(null);
    setSuccess(null);
    setVerificationCodeDebug(null);
    setVerificationCode("");
    setSendVerificationBusy(false);
    setPendingVerificationEmail(null);
    setVerificationFlow(null);
    setVerifiedSignupEmail(null);
    setIsVerificationModalOpen(false);
    setShowChangeEmail(false);
    setChangeEmailPassword("");
    authSubmitInFlightRef.current = false;
    const fadeMs = getAuthModeFadeMs();
    pendingAuthModeFocusRef.current = nextMode;
    setAuthModeTransitioning(fadeMs > 0);
    setAuthMode(nextMode);
    onAuthModeChange?.(nextMode);
    if (fadeMs === 0) {
      authModeTransitionTimerRef.current = null;
      return;
    }
    authModeTransitionTimerRef.current = window.setTimeout(() => {
      setAuthModeTransitioning(false);
      authModeTransitionTimerRef.current = null;
    }, fadeMs);
  }, [authMode, authModeTransitioning, onAuthModeChange]);

  const verificationDigits = Array.from({ length: VERIFICATION_CODE_LENGTH }, (_, index) => verificationCode[index] ?? "");

  const focusVerificationDigit = useCallback((index: number) => {
    const target = verificationDigitInputRefs.current[index];
    if (!target) {
      return;
    }
    target.focus();
    target.select();
  }, []);

  const normalizeVerificationDigits = useCallback((value: string) => value.replace(/\D/g, "").slice(0, VERIFICATION_CODE_LENGTH), []);

  const onVerificationDigitChange = useCallback((index: number, event: ChangeEvent<HTMLInputElement>) => {
    const nextDigits = normalizeVerificationDigits(event.target.value);
    if (!nextDigits) {
      setVerificationCode((prev) => prev.slice(0, index));
      return;
    }
    setVerificationCode((prev) => `${normalizeVerificationDigits(prev).slice(0, index)}${nextDigits}`.slice(0, VERIFICATION_CODE_LENGTH));
    const nextFocusIndex = Math.min(index + nextDigits.length, VERIFICATION_CODE_LENGTH - 1);
    window.requestAnimationFrame(() => {
      focusVerificationDigit(nextFocusIndex);
    });
  }, [focusVerificationDigit, normalizeVerificationDigits]);

  const onVerificationDigitKeyDown = useCallback((index: number, event: ReactKeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Backspace" && !verificationDigits[index] && index > 0) {
      event.preventDefault();
      setVerificationCode((prev) => prev.slice(0, Math.max(0, index - 1)));
      focusVerificationDigit(index - 1);
      return;
    }
    if (event.key === "ArrowLeft" && index > 0) {
      event.preventDefault();
      focusVerificationDigit(index - 1);
      return;
    }
    if (event.key === "ArrowRight" && index < VERIFICATION_CODE_LENGTH - 1) {
      event.preventDefault();
      focusVerificationDigit(index + 1);
    }
  }, [focusVerificationDigit, verificationDigits]);

  const onVerificationDigitPaste = useCallback((event: ReactClipboardEvent<HTMLInputElement>) => {
    const pastedDigits = normalizeVerificationDigits(event.clipboardData.getData("text"));
    if (!pastedDigits) {
      return;
    }
    event.preventDefault();
    setVerificationCode(pastedDigits);
    window.requestAnimationFrame(() => {
      focusVerificationDigit(Math.min(pastedDigits.length, VERIFICATION_CODE_LENGTH - 1));
    });
  }, [focusVerificationDigit, normalizeVerificationDigits]);

  const backLabel = displayView === "default" ? "Back to ReelAI" : "Back to Account Manager";

  return (
    <main className="fixed inset-0 h-[100dvh] w-screen overflow-y-auto bg-black text-white">
      <div
        data-top-chrome="account-back"
        aria-hidden={isVerificationModalOpen}
        inert={isVerificationModalOpen}
        className="top-nav-fade fixed inset-x-0 top-0 z-20 px-4 pb-2 pt-[calc(env(safe-area-inset-top)+1rem)] sm:px-6"
      >
        <button
          type="button"
          onClick={onBack}
          className="inline-flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-medium text-white/72 transition-colors hover:bg-white/[0.07] hover:text-white"
        >
          <i className="fa-solid fa-arrow-left text-[10px]" aria-hidden="true" />
          {backLabel}
        </button>
      </div>
      <div
        aria-hidden={isVerificationModalOpen}
        inert={isVerificationModalOpen}
        className="relative z-10 flex min-h-full w-full items-center justify-center px-5 py-24 sm:px-8"
      >
        <section className="flex w-full max-w-[380px] flex-col bg-transparent">
          <p aria-label="ReelAI" className="mb-10 text-center text-[28px] font-semibold tracking-[-0.04em] text-white">
            ReelAI
          </p>

                {account ? (
                  displayView === "change-password" ? (
                    <div
                      className={`flex flex-1 flex-col justify-center pb-6 transition-opacity duration-[420ms] ease-in-out motion-reduce:transition-none lg:pb-0 ${
                        authContentVisible ? "opacity-100" : "pointer-events-none opacity-0"
                      }`}
                    >
                      <h1 className="text-[2.3rem] font-semibold leading-[1.05] tracking-[-0.04em] text-white sm:text-[2.7rem]">
                        Change Password
                      </h1>
                      <p className="mt-4 text-sm leading-6 text-white">
                        Enter your old password and choose a new one for this account.
                      </p>

                      <div className="mt-7 rounded-[24px] bg-[#202020] px-5 py-5">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-white">Active Session</p>
                        <p className="mt-3 text-2xl font-semibold tracking-[-0.03em] text-white">@{account.username}</p>
                        {account.email ? <p className="mt-2 text-sm text-white">{account.email}</p> : null}
                      </div>

                      <form onSubmit={onChangePassword} className="mt-6 flex flex-col gap-3">
                        <input
                          type="password"
                          value={currentPassword}
                          onChange={(event) => setCurrentPassword(event.target.value)}
                          autoComplete="current-password"
                          placeholder="Old password"
                          className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 text-sm text-white outline-none placeholder:text-white placeholder:opacity-100"
                        />
                        <input
                          type="password"
                          value={newPassword}
                          onChange={(event) => setNewPassword(event.target.value)}
                          autoComplete="new-password"
                          placeholder="New password"
                          className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 text-sm text-white outline-none placeholder:text-white placeholder:opacity-100"
                        />
                        {changePasswordError ? <p className="text-sm text-[#ffb4b4]">{changePasswordError}</p> : null}
                        {changePasswordSuccess ? <p className="text-sm text-[#9ef8cb]">{changePasswordSuccess}</p> : null}
                        <button
                          type="submit"
                          disabled={changePasswordBusy}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition-colors hover:bg-white/[0.07] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
                        >
                          {changePasswordBusy ? "Changing..." : "Change Password"}
                        </button>
                      </form>
                    </div>
                  ) : displayView === "delete-account" ? (
                    <div
                      className={`flex flex-1 flex-col justify-center pb-6 transition-opacity duration-[420ms] ease-in-out motion-reduce:transition-none lg:pb-0 ${
                        authContentVisible ? "opacity-100" : "pointer-events-none opacity-0"
                      }`}
                    >
                      <h1 className="text-[2.3rem] font-semibold leading-[1.05] tracking-[-0.04em] text-white sm:text-[2.7rem]">
                        Delete Account
                      </h1>
                      <p className="mt-4 text-sm leading-6 text-white">
                        Permanently delete this account, its signed-in sessions, saved history, settings, and your sets.
                      </p>

                      <div className="mt-7 rounded-[24px] bg-[#202020] px-5 py-5">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-white">Active Session</p>
                        <p className="mt-3 text-2xl font-semibold tracking-[-0.03em] text-white">@{account.username}</p>
                        {account.email ? <p className="mt-2 text-sm text-white">{account.email}</p> : null}
                      </div>

                      <form onSubmit={onDeleteAccount} className="mt-6 flex flex-col gap-3">
                        <p className="text-sm leading-6 text-[#ffb4b4]">
                          This action cannot be undone.
                        </p>
                        <p className="text-xs leading-5 text-white/65">
                          Any active Stripe subscription is canceled automatically when your account is deleted.
                        </p>
                        <input
                          type="password"
                          value={deleteAccountPassword}
                          onChange={(event) => setDeleteAccountPassword(event.target.value)}
                          autoComplete="current-password"
                          placeholder="Current password"
                          className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 text-sm text-white outline-none placeholder:text-white placeholder:opacity-100"
                        />
                        {deleteAccountError ? <p className="text-sm text-[#ffb4b4]">{deleteAccountError}</p> : null}
                        <button
                          type="submit"
                          disabled={deleteAccountBusy}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#8f232a] px-5 text-sm font-semibold text-white transition hover:bg-[#a12831] disabled:cursor-not-allowed disabled:bg-[#8f232a]/45 disabled:text-white/35"
                        >
                          {deleteAccountBusy ? "Deleting..." : "Delete Account"}
                        </button>
                      </form>
                    </div>
                  ) : account.isVerified ? (
                    <div
                      className={`flex flex-1 flex-col justify-start overflow-y-auto py-4 transition-opacity duration-300 motion-reduce:transition-none lg:py-2 ${
                        authContentVisible ? "opacity-100" : "pointer-events-none opacity-0"
                      }`}
                    >
                      <h1 className="text-[2.3rem] font-semibold leading-[1.05] tracking-[-0.04em] text-white sm:text-[2.7rem]">
                        You&apos;re signed in
                      </h1>
                      <p className="mt-4 text-sm leading-6 text-white">
                        This account can open and manage your sets from any device that signs in with the same username and password.
                      </p>

                      <div className="mt-7 rounded-[24px] bg-[#202020] px-5 py-5">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-white">Active Session</p>
                        <p className="mt-3 text-2xl font-semibold tracking-[-0.03em] text-white">@{account.username}</p>
                        {account.email ? <p className="mt-2 text-sm text-white">{account.email}</p> : null}
                      </div>

                      <BillingPlanUsageCard account={account} />

                      {error ? <p className="mt-4 text-sm text-[#ffb4b4]">{error}</p> : null}
                      {success ? <p className="mt-4 text-sm text-[#9ef8cb]">{success}</p> : null}

                      <div className="mt-4 flex flex-col gap-3">
                        <button
                          type="button"
                          onClick={onOpenChangePassword}
                          disabled={busy || deleteAccountBusy}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition-colors hover:bg-white/[0.07] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
                        >
                          Change Password
                        </button>
                        <button
                          type="button"
                          onClick={onSignOut}
                          disabled={busy || deleteAccountBusy}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition-colors hover:bg-white/[0.07] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
                        >
                          {busy ? "Signing out..." : "Sign Out"}
                        </button>
                        <button
                          type="button"
                          onClick={onOpenDeleteAccount}
                          disabled={busy || deleteAccountBusy}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#622329] px-5 text-sm font-semibold text-white transition hover:bg-[#7a2b33] disabled:cursor-not-allowed disabled:bg-[#622329]/45 disabled:text-white/35"
                        >
                          Delete Account
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div
                      className={`flex flex-1 flex-col justify-center pb-6 transition-opacity duration-[420ms] ease-in-out motion-reduce:transition-none lg:pb-0 ${
                        authContentVisible ? "opacity-100" : "pointer-events-none opacity-0"
                      }`}
                    >
                      <h1 className="text-[2.3rem] font-semibold leading-[1.05] tracking-[-0.04em] text-white sm:text-[2.7rem]">
                        Verify your account
                      </h1>
                      <p className="mt-4 text-sm leading-6 text-white">
                        Confirm the email attached to this account before you can open or edit Your Sets.
                      </p>

                      <div className="mt-7 rounded-[24px] bg-[#202020] px-5 py-5">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-white">Pending Verification</p>
                        <p className="mt-3 text-2xl font-semibold tracking-[-0.03em] text-white">@{account.username}</p>
                        {account.email ? <p className="mt-2 text-sm text-white">{account.email}</p> : null}
                        <p className="mt-3 text-sm leading-6 text-white">
                          Enter the latest code we sent you. The code expires after 20 minutes.
                        </p>
                      </div>

                      <form onSubmit={onVerifyAccount} className="mt-6 flex flex-col gap-3">
                        <input
                          value={verificationCode}
                          onChange={(event) => setVerificationCode(event.target.value)}
                          inputMode="numeric"
                          autoComplete="one-time-code"
                          placeholder="Verification code"
                          className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 text-sm text-white outline-none placeholder:text-white placeholder:opacity-100"
                        />
                        {verificationCodeDebug ? (
                          <p className="text-sm text-[#f6d38b]">Local debug code: {verificationCodeDebug}</p>
                        ) : null}
                        {error ? <p className="text-sm text-[#ffb4b4]">{error}</p> : null}
                        {success ? <p className="text-sm text-[#9ef8cb]">{success}</p> : null}
                        <button
                          type="submit"
                          disabled={verificationBusy}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition-colors hover:bg-white/[0.07] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
                        >
                          {verificationBusy ? "Verifying..." : "Verify Account"}
                        </button>
                      </form>

                      <div className="mt-4 flex flex-col gap-3">
                        <button
                          type="button"
                          onClick={onResendVerification}
                          disabled={resendBusy || changeEmailBusy}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition-colors hover:bg-white/[0.07] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
                        >
                          {resendBusy ? "Sending..." : "Resend Code"}
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setShowChangeEmail((prev) => !prev);
                            setChangeEmailValue(account.email ?? "");
                            setChangeEmailPassword("");
                            setError(null);
                            setSuccess(null);
                          }}
                          disabled={changeEmailBusy || resendBusy}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition-colors hover:bg-white/[0.07] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
                        >
                          {showChangeEmail ? "Cancel Email Change" : "Change Email"}
                        </button>
                        {showChangeEmail ? (
                          <form onSubmit={onChangeVerificationEmail} className="mt-1 flex flex-col gap-3">
                            <input
                              type="email"
                              value={changeEmailValue}
                              onChange={(event) => setChangeEmailValue(event.target.value)}
                              autoComplete="email"
                              placeholder="New email"
                              className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 text-sm text-white outline-none placeholder:text-white placeholder:opacity-100"
                            />
                            <input
                              type="password"
                              value={changeEmailPassword}
                              onChange={(event) => setChangeEmailPassword(event.target.value)}
                              autoComplete="current-password"
                              placeholder="Current password"
                              className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 text-sm text-white outline-none placeholder:text-white placeholder:opacity-100"
                            />
                            <button
                              type="submit"
                              disabled={changeEmailBusy}
                              className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition-colors hover:bg-white/[0.07] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
                            >
                              {changeEmailBusy ? "Updating..." : "Update Email"}
                            </button>
                          </form>
                        ) : null}
                        <button
                          type="button"
                          onClick={onSignOut}
                          disabled={busy || verificationBusy || resendBusy || changeEmailBusy}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition-colors hover:bg-white/[0.07] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
                        >
                          {busy ? "Signing out..." : "Sign Out"}
                        </button>
                      </div>
                    </div>
                  )
                ) : (
                  <div className="grid w-full">
                    {(["login", "register"] as const).map((mode) => {
                      const isActiveMode = authMode === mode;
                      const isRegister = mode === "register";
                      return (
                        <div
                          key={mode}
                          data-auth-mode={mode}
                          aria-hidden={!isActiveMode}
                          inert={!isActiveMode || authModeTransitioning}
                          className={`col-start-1 row-start-1 flex flex-col justify-center pb-6 transition-opacity duration-[420ms] ease-in-out [will-change:opacity] motion-reduce:transition-none lg:pb-0 ${
                            isActiveMode ? (authModeTransitioning ? "pointer-events-none opacity-100" : "opacity-100") : "pointer-events-none opacity-0"
                          }`}
                        >
                          <h1 className="text-center text-[2rem] font-semibold leading-tight tracking-[-0.04em] text-white sm:text-[2.2rem]">
                            {isRegister ? "Create your account" : "Welcome back"}
                          </h1>
                          <p className="mt-3 text-center text-sm leading-6 text-white/55">
                            {isRegister
                              ? "Sign up to save searches and build your reel library."
                              : "Sign in to continue to ReelAI."}
                          </p>

                          <form onSubmit={onSubmit} className="mt-8 flex flex-col gap-4">
                            <label className="block">
                              <span className="mb-1.5 block text-sm font-medium text-white">Username</span>
                              <input
                                ref={(node) => {
                                  authModeFirstFieldRefs.current[mode] = node;
                                }}
                                value={username}
                                onChange={(event) => setUsername(event.target.value)}
                                autoComplete="username"
                                placeholder="Username"
                                className="community-auth-input h-12 w-full rounded-[10px] bg-[#202020] px-4 text-sm text-white outline-none placeholder:text-white/35 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                              />
                            </label>
                            {isRegister ? (
                              <label className="block">
                                <span className="mb-1.5 block text-sm font-medium text-white">Email</span>
                                <div className="relative">
                                  <input
                                    type="email"
                                    value={email}
                                    onChange={onSignupEmailChange}
                                    autoComplete="email"
                                    placeholder="you@example.com"
                                    className="community-auth-input h-12 w-full rounded-[10px] bg-[#202020] px-4 pr-[5.75rem] text-sm text-white outline-none placeholder:text-white/35 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                                  />
                                  <button
                                    type="button"
                                    onClick={() => {
                                      void onSendVerificationEmail();
                                    }}
                                    disabled={busy || sendVerificationBusy || !email.trim()}
                                    className="absolute right-1.5 top-1/2 inline-flex h-9 -translate-y-1/2 items-center justify-center rounded-lg bg-black px-3 text-[11px] font-semibold uppercase tracking-[0.08em] text-white transition-colors duration-300 enabled:hover:bg-white/[0.07] disabled:cursor-not-allowed disabled:bg-black/35 disabled:text-white/40 disabled:transition-none"
                                  >
                                    Send
                                  </button>
                                </div>
                              </label>
                            ) : null}
                            <label className="block">
                              <span className="mb-1.5 block text-sm font-medium text-white">Password</span>
                              <input
                                type="password"
                                value={password}
                                onChange={(event) => setPassword(event.target.value)}
                                autoComplete={isRegister ? "new-password" : "current-password"}
                                placeholder={isRegister ? "At least 8 characters" : "Enter your password"}
                                className="community-auth-input h-12 w-full rounded-[10px] bg-[#202020] px-4 text-sm text-white outline-none placeholder:text-white/35 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                              />
                            </label>
                            <div className="space-y-2">
                              {error ? <p className="text-sm text-[#ffb4b4]">{error}</p> : null}
                              {success ? <p className="text-sm text-[#9ef8cb]">{success}</p> : null}
                            </div>
                            <button
                              type="submit"
                              disabled={busy || sendVerificationBusy}
                              className="inline-flex h-12 w-full items-center justify-center rounded-[10px] bg-white px-5 text-sm font-semibold text-black transition-colors duration-300 hover:bg-white/90 disabled:cursor-not-allowed disabled:bg-white/20 disabled:text-white/35"
                            >
                              {busy
                                ? isRegister
                                  ? "Creating account..."
                                  : "Signing in..."
                                : isRegister
                                  ? "Create Account"
                                  : "Login"}
                            </button>
                          </form>

                          <div className="mt-7 text-center text-sm text-white/55">
                            {isRegister ? "Already have an account?" : "Need an account?"}
                            {" "}
                            <button
                              type="button"
                              onClick={() => switchAuthMode(isRegister ? "login" : "register")}
                              disabled={authModeTransitioning}
                              className="ml-1 px-1 py-1 font-semibold text-white transition-opacity hover:opacity-70 disabled:cursor-not-allowed disabled:opacity-45"
                            >
                              {isRegister ? "Sign in" : "Create one"}
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
        </section>
      </div>
      <FadePresence show={isVerificationModalOpen}>
        {(modalVisible) => isVerificationModalOpen ? (
        <ViewportModalPortal>
          <div
            className={`fixed inset-0 z-[170] flex items-center justify-center bg-black/80 px-4 py-6 transition-opacity duration-300 motion-reduce:transition-none ${
              modalVisible ? "opacity-100" : "opacity-0"
            }`}
            role="presentation"
            onClick={closeVerificationModal}
            >
            <div
              ref={verificationDialogRef}
              role="dialog"
              aria-modal="true"
              aria-label="Verify account"
              tabIndex={-1}
              className="w-full max-w-md rounded-[14px] bg-[#202020] p-5 text-white sm:p-6"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-center justify-between gap-3">
                <button
                  type="button"
                  onClick={closeVerificationModal}
                  disabled={verificationBusy}
                  className="inline-flex items-center gap-2 rounded-full px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-white/72 transition-colors hover:bg-white/[0.07] hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <i className="fa-solid fa-arrow-left text-[10px]" aria-hidden="true" />
                  Return
                </button>
              </div>

              <div className="mt-5">
                <h2 className="text-2xl font-semibold tracking-[-0.04em] text-white">Enter verification code</h2>
              </div>

              <form onSubmit={onVerifyAccount} className="mt-6">
                <div className="grid grid-cols-6 gap-2 sm:gap-3">
                  {verificationDigits.map((digit, index) => (
                    <input
                      key={`verification-digit-${index}`}
                      ref={(node) => {
                        verificationDigitInputRefs.current[index] = node;
                      }}
                      value={digit}
                      onChange={(event) => onVerificationDigitChange(index, event)}
                      onKeyDown={(event) => onVerificationDigitKeyDown(index, event)}
                      onPaste={onVerificationDigitPaste}
                      inputMode="numeric"
                      autoComplete={index === 0 ? "one-time-code" : "off"}
                      aria-label={`Verification digit ${index + 1}`}
                      maxLength={VERIFICATION_CODE_LENGTH}
                      className="h-14 w-full rounded-[14px] bg-[#2c2c2c] text-center text-xl font-semibold text-white outline-none transition focus:bg-[#343434]"
                    />
                  ))}
                </div>

                {verificationCodeDebug ? (
                  <p className="mt-4 text-sm text-[#f6d38b]">Local debug code: {verificationCodeDebug}</p>
                ) : null}
                {error ? <p className="mt-4 text-sm text-[#ffb4b4]">{error}</p> : null}
                {success ? <p className="mt-4 text-sm text-[#9ef8cb]">{success}</p> : null}

                <button
                  type="submit"
                  disabled={verificationBusy || verificationCode.trim().length !== VERIFICATION_CODE_LENGTH}
                  className="mt-5 inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition-colors hover:bg-white/[0.07] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
                >
                  {verificationBusy ? "Verifying..." : verificationFlow === "signup" ? "Verify Email" : "Verify"}
                </button>
              </form>
            </div>
          </div>
        </ViewportModalPortal>
        ) : null}
      </FadePresence>
    </main>
  );
}
