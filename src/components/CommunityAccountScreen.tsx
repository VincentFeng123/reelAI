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
import { ViewportModalPortal } from "@/components/ViewportModalPortal";
import type { CommunityAccount } from "@/lib/types";

type CommunityAccountScreenProps = {
  account: CommunityAccount | null;
  view: "default" | "change-password" | "delete-account";
  onBack: () => void;
  onAccountChange: (account: CommunityAccount | null) => void;
  onOpenChangePassword: () => void;
  onOpenDeleteAccount: () => void;
};

const AUTH_MODE_FADE_MS = 160;
const VERIFICATION_CODE_LENGTH = 6;

function normalizeSignupEmailForComparison(value: string): string {
  return value.trim().toLowerCase();
}

export function CommunityAccountScreen({
  account,
  view,
  onBack,
  onAccountChange,
  onOpenChangePassword,
  onOpenDeleteAccount,
}: CommunityAccountScreenProps) {
  const [authMode, setAuthMode] = useState<"register" | "login">("login");
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

  const closeVerificationModal = useCallback(() => {
    if (verificationBusy) {
      return;
    }
    setIsVerificationModalOpen(false);
    if (!account) {
      setVerificationFlow(null);
    }
  }, [account, verificationBusy]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && isVerificationModalOpen && !verificationBusy) {
        event.preventDefault();
        closeVerificationModal();
        return;
      }
      if (event.key === "Escape" && !busy && !sendVerificationBusy && !verificationBusy && !resendBusy && !changeEmailBusy && !deleteAccountBusy) {
        onBack();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
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
    };
  }, []);

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
    setAuthModeTransitioning(true);
    setAuthContentVisible(false);
    authModeTransitionTimerRef.current = window.setTimeout(() => {
      setDisplayView(view);
      window.requestAnimationFrame(() => {
        setAuthContentVisible(true);
        authModeTransitionTimerRef.current = window.setTimeout(() => {
          setAuthModeTransitioning(false);
          authModeTransitionTimerRef.current = null;
        }, AUTH_MODE_FADE_MS);
      });
    }, AUTH_MODE_FADE_MS);
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
        setIsVerificationModalOpen(true);
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
  }, [authMode, busy, email, onAccountChange, password, sendVerificationBusy, username, verifiedSignupEmail]);

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
        setIsVerificationModalOpen(true);
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
      setIsVerificationModalOpen(true);
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
      setIsVerificationModalOpen(true);
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

  const isRegister = authMode === "register";
  const switchAuthMode = useCallback((nextMode: "register" | "login") => {
    if (nextMode === authMode || authModeTransitioning) {
      return;
    }
    if (authModeTransitionTimerRef.current !== null) {
      window.clearTimeout(authModeTransitionTimerRef.current);
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
    setAuthModeTransitioning(true);
    setAuthContentVisible(false);
    authModeTransitionTimerRef.current = window.setTimeout(() => {
      setAuthMode(nextMode);
      window.requestAnimationFrame(() => {
        setAuthContentVisible(true);
        authModeTransitionTimerRef.current = window.setTimeout(() => {
          setAuthModeTransitioning(false);
          authModeTransitionTimerRef.current = null;
        }, AUTH_MODE_FADE_MS);
      });
    }, AUTH_MODE_FADE_MS);
  }, [authMode, authModeTransitioning]);

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
    <main className="fixed inset-0 h-[100dvh] w-screen overflow-hidden bg-black text-black">
      <div className="absolute inset-0">
        <img
          src="/images/community/80543.jpg"
          alt=""
          aria-hidden="true"
          className="h-full w-full scale-[1.04] rotate-180 object-cover opacity-[0.14]"
        />
        <div className="absolute inset-0 bg-black/46" />
      </div>

      <div className="fixed left-0 top-0 z-20 px-4 pb-2 pt-[calc(env(safe-area-inset-top)+1rem)] sm:px-6 lg:hidden">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex items-center gap-2 rounded-full px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-white transition-colors hover:bg-white/10"
        >
          <i className="fa-solid fa-arrow-left text-[10px]" aria-hidden="true" />
          {backLabel}
        </button>
      </div>

      <div className="relative z-10 h-full w-full lg:flex lg:items-center lg:justify-center lg:px-10 lg:py-8">
        <div className="h-full w-full lg:h-auto lg:max-w-[1080px] xl:max-w-[1120px]">
          <div className="h-full overflow-hidden bg-white/[0.07] backdrop-blur-[18px] backdrop-saturate-150 lg:h-auto lg:rounded-[32px] lg:shadow-[0_32px_140px_rgba(10,5,20,0.38)]">
            <div className="grid h-full w-full lg:h-[min(82dvh,680px)] lg:grid-cols-[minmax(0,0.94fr)_minmax(320px,0.96fr)]">
            <section className="order-2 flex min-h-full bg-transparent px-6 pb-8 pt-24 sm:px-10 sm:pt-28 lg:order-1 lg:px-12 lg:py-10 xl:px-14 xl:py-12">
              <div className="mx-auto flex h-full w-full max-w-[360px] flex-col">
                <div className="shrink-0">
                  <button
                    type="button"
                    onClick={onBack}
                    className="hidden items-center gap-2 rounded-full px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-white transition-colors hover:bg-white/10 lg:inline-flex"
                  >
                    <i className="fa-solid fa-arrow-left text-[10px]" aria-hidden="true" />
                    {backLabel}
                  </button>
                </div>

                {account ? (
                  displayView === "change-password" ? (
                    <div
                      className={`flex flex-1 flex-col justify-center pb-6 transition-opacity duration-150 lg:pb-0 ${
                        authContentVisible ? "opacity-100" : "pointer-events-none opacity-0"
                      }`}
                    >
                      <h1 className="text-[2.3rem] font-semibold leading-[1.05] tracking-[-0.04em] text-white sm:text-[2.7rem]">
                        Change Password
                      </h1>
                      <p className="mt-4 text-sm leading-6 text-white">
                        Enter your old password and choose a new one for this account.
                      </p>

                      <div className="mt-7 rounded-[24px] bg-black/22 px-5 py-5 shadow-[0_16px_40px_rgba(16,16,16,0.16)] backdrop-blur-[16px]">
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
                          className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 text-sm text-white outline-none placeholder:text-white placeholder:opacity-100 backdrop-blur-[10px]"
                        />
                        <input
                          type="password"
                          value={newPassword}
                          onChange={(event) => setNewPassword(event.target.value)}
                          autoComplete="new-password"
                          placeholder="New password"
                          className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 text-sm text-white outline-none placeholder:text-white placeholder:opacity-100 backdrop-blur-[10px]"
                        />
                        {changePasswordError ? <p className="text-sm text-[#ffb4b4]">{changePasswordError}</p> : null}
                        {changePasswordSuccess ? <p className="text-sm text-[#9ef8cb]">{changePasswordSuccess}</p> : null}
                        <button
                          type="submit"
                          disabled={changePasswordBusy}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition hover:bg-[#2a2a2a] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
                        >
                          {changePasswordBusy ? "Changing..." : "Change Password"}
                        </button>
                      </form>
                    </div>
                  ) : displayView === "delete-account" ? (
                    <div
                      className={`flex flex-1 flex-col justify-center pb-6 transition-opacity duration-150 lg:pb-0 ${
                        authContentVisible ? "opacity-100" : "pointer-events-none opacity-0"
                      }`}
                    >
                      <h1 className="text-[2.3rem] font-semibold leading-[1.05] tracking-[-0.04em] text-white sm:text-[2.7rem]">
                        Delete Account
                      </h1>
                      <p className="mt-4 text-sm leading-6 text-white">
                        Permanently delete this account, its signed-in sessions, saved history, settings, and your sets.
                      </p>

                      <div className="mt-7 rounded-[24px] bg-black/22 px-5 py-5 shadow-[0_16px_40px_rgba(16,16,16,0.16)] backdrop-blur-[16px]">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-white">Active Session</p>
                        <p className="mt-3 text-2xl font-semibold tracking-[-0.03em] text-white">@{account.username}</p>
                        {account.email ? <p className="mt-2 text-sm text-white">{account.email}</p> : null}
                      </div>

                      <form onSubmit={onDeleteAccount} className="mt-6 flex flex-col gap-3">
                        <p className="text-sm leading-6 text-[#ffb4b4]">
                          This action cannot be undone.
                        </p>
                        <input
                          type="password"
                          value={deleteAccountPassword}
                          onChange={(event) => setDeleteAccountPassword(event.target.value)}
                          autoComplete="current-password"
                          placeholder="Current password"
                          className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 text-sm text-white outline-none placeholder:text-white placeholder:opacity-100 backdrop-blur-[10px]"
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
                      className={`flex flex-1 flex-col justify-center pb-6 transition-opacity duration-150 lg:pb-0 ${
                        authContentVisible ? "opacity-100" : "pointer-events-none opacity-0"
                      }`}
                    >
                      <h1 className="text-[2.3rem] font-semibold leading-[1.05] tracking-[-0.04em] text-white sm:text-[2.7rem]">
                        You&apos;re signed in
                      </h1>
                      <p className="mt-4 text-sm leading-6 text-white">
                        This account can open and manage your sets from any device that signs in with the same username and password.
                      </p>

                      <div className="mt-7 rounded-[24px] bg-black/22 px-5 py-5 shadow-[0_16px_40px_rgba(16,16,16,0.16)] backdrop-blur-[16px]">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-white">Active Session</p>
                        <p className="mt-3 text-2xl font-semibold tracking-[-0.03em] text-white">@{account.username}</p>
                        {account.email ? <p className="mt-2 text-sm text-white">{account.email}</p> : null}
                      </div>

                      {error ? <p className="mt-4 text-sm text-[#ffb4b4]">{error}</p> : null}
                      {success ? <p className="mt-4 text-sm text-[#9ef8cb]">{success}</p> : null}

                      <div className="mt-7 flex flex-col gap-3">
                        <button
                          type="button"
                          onClick={onOpenChangePassword}
                          disabled={busy || deleteAccountBusy}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition hover:bg-[#2a2a2a] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
                        >
                          Change Password
                        </button>
                        <button
                          type="button"
                          onClick={onSignOut}
                          disabled={busy || deleteAccountBusy}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition hover:bg-[#2a2a2a] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
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
                      className={`flex flex-1 flex-col justify-center pb-6 transition-opacity duration-150 lg:pb-0 ${
                        authContentVisible ? "opacity-100" : "pointer-events-none opacity-0"
                      }`}
                    >
                      <h1 className="text-[2.3rem] font-semibold leading-[1.05] tracking-[-0.04em] text-white sm:text-[2.7rem]">
                        Verify your account
                      </h1>
                      <p className="mt-4 text-sm leading-6 text-white">
                        Confirm the email attached to this account before you can open or edit Your Sets.
                      </p>

                      <div className="mt-7 rounded-[24px] bg-black/22 px-5 py-5 shadow-[0_16px_40px_rgba(16,16,16,0.16)] backdrop-blur-[16px]">
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
                          className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 text-sm text-white outline-none placeholder:text-white placeholder:opacity-100 backdrop-blur-[10px]"
                        />
                        {verificationCodeDebug ? (
                          <p className="text-sm text-[#f6d38b]">Local debug code: {verificationCodeDebug}</p>
                        ) : null}
                        {error ? <p className="text-sm text-[#ffb4b4]">{error}</p> : null}
                        {success ? <p className="text-sm text-[#9ef8cb]">{success}</p> : null}
                        <button
                          type="submit"
                          disabled={verificationBusy}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition hover:bg-[#2a2a2a] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
                        >
                          {verificationBusy ? "Verifying..." : "Verify Account"}
                        </button>
                      </form>

                      <div className="mt-4 flex flex-col gap-3">
                        <button
                          type="button"
                          onClick={onResendVerification}
                          disabled={resendBusy || changeEmailBusy}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition hover:bg-[#2a2a2a] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
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
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition hover:bg-[#2a2a2a] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
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
                              className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 text-sm text-white outline-none placeholder:text-white placeholder:opacity-100 backdrop-blur-[10px]"
                            />
                            <input
                              type="password"
                              value={changeEmailPassword}
                              onChange={(event) => setChangeEmailPassword(event.target.value)}
                              autoComplete="current-password"
                              placeholder="Current password"
                              className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 text-sm text-white outline-none placeholder:text-white placeholder:opacity-100 backdrop-blur-[10px]"
                            />
                            <button
                              type="submit"
                              disabled={changeEmailBusy}
                              className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition hover:bg-[#2a2a2a] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
                            >
                              {changeEmailBusy ? "Updating..." : "Update Email"}
                            </button>
                          </form>
                        ) : null}
                        <button
                          type="button"
                          onClick={onSignOut}
                          disabled={busy || verificationBusy || resendBusy || changeEmailBusy}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition hover:bg-[#2a2a2a] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
                        >
                          {busy ? "Signing out..." : "Sign Out"}
                        </button>
                      </div>
                    </div>
                  )
                ) : (
                  <div
                    className={`flex flex-1 flex-col justify-center pb-6 transition-opacity duration-150 lg:pb-0 ${
                      authContentVisible ? "opacity-100" : "pointer-events-none opacity-0"
                    }`}
                  >
                    <h1 className="text-[2.3rem] font-semibold leading-[1.05] tracking-[-0.05em] text-white sm:text-[2.8rem]">
                      {isRegister ? "Create account" : "Login"}
                    </h1>
                    <p className="mt-3 text-sm leading-6 text-white">
                      {isRegister
                        ? "Create credentials for your private cross-device reel library."
                        : "Enter your credentials to get in."}
                    </p>

                    <form onSubmit={onSubmit} className="mt-5 flex flex-col gap-4">
                      <label className="block">
                        <span className="mb-1.5 block text-sm font-medium text-white">Username</span>
                        <input
                          value={username}
                          onChange={(event) => setUsername(event.target.value)}
                          autoComplete="username"
                          placeholder="Username"
                          className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 text-sm text-white outline-none placeholder:text-white placeholder:opacity-100 backdrop-blur-[10px]"
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
                              className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 pr-[5.75rem] text-sm text-white outline-none placeholder:text-white placeholder:opacity-100 backdrop-blur-[10px]"
                            />
                            <button
                              type="button"
                              onClick={() => {
                                void onSendVerificationEmail();
                              }}
                              disabled={busy || sendVerificationBusy || !email.trim()}
                              className="absolute right-1.5 top-1/2 inline-flex h-9 -translate-y-1/2 items-center justify-center rounded-[10px] bg-black px-3 text-[11px] font-semibold uppercase tracking-[0.08em] text-white transition-colors duration-150 enabled:hover:bg-[#181818] disabled:cursor-not-allowed disabled:bg-black/35 disabled:text-white/40 disabled:transition-none"
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
                          className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 text-sm text-white outline-none placeholder:text-white placeholder:opacity-100 backdrop-blur-[10px]"
                        />
                      </label>
                      <div className="space-y-2">
                        {error ? <p className="text-sm text-[#ffb4b4]">{error}</p> : null}
                        {success ? <p className="text-sm text-[#9ef8cb]">{success}</p> : null}
                      </div>
                      <button
                        type="submit"
                        disabled={busy || sendVerificationBusy}
                        className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition hover:bg-[#2a2a2a] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45"
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

                    <div className="mt-7 text-center text-sm text-white">
                      {isRegister ? "Already have an account?" : "Need an account?"}
                      {" "}
                      <button
                        type="button"
                        onClick={() => switchAuthMode(isRegister ? "login" : "register")}
                        disabled={authModeTransitioning}
                        className="font-semibold text-white transition hover:text-white"
                      >
                        {isRegister ? "Sign in" : "Create one"}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </section>

            <section className="relative hidden bg-transparent lg:order-2 lg:block lg:h-full lg:p-5">
              <div className="relative h-full min-h-[260px] overflow-hidden rounded-[28px] bg-[#170f24] sm:min-h-[320px]">
                <img
                  src="/images/community/80543.jpg"
                  alt=""
                  aria-hidden="true"
                  className="absolute inset-0 h-full w-full object-cover"
                />
                <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(18,9,28,0.08)_0%,rgba(18,9,28,0.2)_38%,rgba(18,9,28,0.72)_100%)]" />
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_center,rgba(255,142,208,0.18),transparent_28%),radial-gradient(circle_at_bottom_left,rgba(106,98,255,0.22),transparent_34%)]" />

                <div className="absolute right-4 top-4 z-10 grid h-11 w-11 place-items-center rounded-full bg-black text-sm font-semibold tracking-[-0.04em] text-white shadow-[0_12px_28px_rgba(0,0,0,0.32)]">
                  R.
                </div>

                <div className="absolute inset-x-0 bottom-0 z-10 px-6 pb-12 pt-24 sm:px-8 sm:pb-14">
                  <p className="max-w-[16rem] text-[2rem] font-medium leading-[1.05] tracking-[-0.05em] text-white sm:text-[2.45rem]">
                    {account ? "Your reel sets stay with you." : "Your reel sets, everywhere you sign in."}
                  </p>
                  <p className="mt-4 max-w-[22rem] text-sm leading-6 text-white">
                    Private by default, synced by account, and ready from any device that uses the same login.
                  </p>
                </div>
              </div>
            </section>
          </div>
        </div>
        </div>
      </div>
      {isVerificationModalOpen ? (
        <ViewportModalPortal>
          <div
            className="fixed inset-0 z-[140] flex items-center justify-center bg-black/72 px-4 py-6 backdrop-blur-[4px]"
            role="presentation"
            onClick={closeVerificationModal}
          >
            <div
              role="dialog"
              aria-modal="true"
              aria-label="Verify account"
              className="w-full max-w-md rounded-[28px] bg-[#111111] p-5 text-white shadow-[0_28px_90px_rgba(0,0,0,0.52)] sm:p-6"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-center justify-between gap-3">
                <button
                  type="button"
                  onClick={closeVerificationModal}
                  disabled={verificationBusy}
                  className="inline-flex items-center gap-2 rounded-full px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-white/72 transition hover:bg-white/10 hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
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
                  className="mt-5 inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition hover:bg-[#2a2a2a] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
                >
                  {verificationBusy ? "Verifying..." : verificationFlow === "signup" ? "Verify Email" : "Verify"}
                </button>
              </form>
            </div>
          </div>
        </ViewportModalPortal>
      ) : null}
    </main>
  );
}
