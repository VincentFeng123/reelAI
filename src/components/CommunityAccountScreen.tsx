"use client";

import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";

import {
  changeCommunityVerificationEmail,
  changeCommunityPassword,
  loginCommunityAccount,
  logoutCommunityAccount,
  registerCommunityAccount,
  resendCommunityVerification,
  verifyCommunityAccount,
} from "@/lib/api";
import type { CommunityAccount } from "@/lib/types";

type CommunityAccountScreenProps = {
  account: CommunityAccount | null;
  onBack: () => void;
  onAccountChange: (account: CommunityAccount | null) => void;
  onOpenYourSets: () => void;
};

const AUTH_MODE_FADE_MS = 160;

export function CommunityAccountScreen({
  account,
  onBack,
  onAccountChange,
  onOpenYourSets,
}: CommunityAccountScreenProps) {
  const [authMode, setAuthMode] = useState<"register" | "login">("login");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [verificationCode, setVerificationCode] = useState("");
  const [verificationCodeDebug, setVerificationCodeDebug] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [verificationBusy, setVerificationBusy] = useState(false);
  const [resendBusy, setResendBusy] = useState(false);
  const [changeEmailBusy, setChangeEmailBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [authContentVisible, setAuthContentVisible] = useState(true);
  const [authModeTransitioning, setAuthModeTransitioning] = useState(false);
  const authModeTransitionTimerRef = useRef<number | null>(null);
  const [showChangePassword, setShowChangePassword] = useState(false);
  const [showChangeEmail, setShowChangeEmail] = useState(false);
  const [changeEmailValue, setChangeEmailValue] = useState("");
  const [changeEmailPassword, setChangeEmailPassword] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [changePasswordBusy, setChangePasswordBusy] = useState(false);
  const [changePasswordError, setChangePasswordError] = useState<string | null>(null);
  const [changePasswordSuccess, setChangePasswordSuccess] = useState<string | null>(null);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy && !verificationBusy && !resendBusy && !changeEmailBusy) {
        onBack();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [busy, changeEmailBusy, onBack, resendBusy, verificationBusy]);

  useEffect(() => {
    setPassword("");
    setError(null);
    setSuccess(null);
    setVerificationCode("");
    setShowChangePassword(false);
    setShowChangeEmail(false);
    setChangeEmailValue(account?.email ?? "");
    setChangeEmailPassword("");
    setCurrentPassword("");
    setNewPassword("");
    setChangePasswordError(null);
    setChangePasswordSuccess(null);
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
  }, [account]);

  useEffect(() => {
    return () => {
      if (authModeTransitionTimerRef.current !== null) {
        window.clearTimeout(authModeTransitionTimerRef.current);
      }
    };
  }, []);

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (busy) {
      return;
    }
    const trimmedUsername = username.trim();
    const trimmedEmail = email.trim();
    if (!trimmedUsername || !password || (authMode === "register" && !trimmedEmail)) {
      setError(authMode === "register" ? "Enter a username, email, and password." : "Enter a username and password.");
      setSuccess(null);
      return;
    }
    setBusy(true);
    setError(null);
    setSuccess(null);
    try {
      const session = authMode === "register"
        ? await registerCommunityAccount({ username: trimmedUsername, email: trimmedEmail, password })
        : await loginCommunityAccount({ username: trimmedUsername, password });
      onAccountChange(session.account);
      setPassword("");
      setVerificationCodeDebug(session.verificationCodeDebug ?? null);
      if (session.verificationRequired || !session.account.isVerified) {
        const targetEmail = session.account.email ?? (authMode === "register" ? trimmedEmail : "");
        setSuccess(
          targetEmail
            ? `${authMode === "register" ? "Account created" : "Signed in"}. Enter the verification code sent to ${targetEmail}.`
            : `${authMode === "register" ? "Account created" : "Signed in"}. Verify your email to unlock Your Sets.`,
        );
      } else if (session.claimedLegacySets > 0) {
        setSuccess(
          `${authMode === "register" ? "Account created" : "Signed in"} and linked ${session.claimedLegacySets} existing set${session.claimedLegacySets === 1 ? "" : "s"} from this device.`,
        );
      } else {
        setSuccess(authMode === "register" ? "Account created." : "Signed in.");
      }
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Could not complete that request.");
      setSuccess(null);
    } finally {
      setBusy(false);
    }
  };

  const onSignOut = async () => {
    if (busy || verificationBusy || resendBusy || changeEmailBusy || changePasswordBusy) {
      return;
    }
    setBusy(true);
    setError(null);
    setSuccess(null);
    setVerificationCodeDebug(null);
    setShowChangeEmail(false);
    setChangeEmailPassword("");
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
      const result = await verifyCommunityAccount({ code: trimmedCode });
      onAccountChange(result.account);
      setVerificationCode("");
      setVerificationCodeDebug(null);
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
      setShowChangeEmail(false);
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
    setShowChangeEmail(false);
    setChangeEmailPassword("");
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

      <div className="relative z-10 h-full w-full lg:flex lg:items-center lg:justify-center lg:px-10 lg:py-8">
        <div className="h-full w-full lg:h-auto lg:max-w-[1080px] xl:max-w-[1120px]">
          <div className="h-full overflow-hidden bg-black/22 backdrop-blur-[22px] lg:h-auto lg:rounded-[32px] lg:border lg:border-[#8c8c95]/45 lg:shadow-[0_32px_140px_rgba(10,5,20,0.42)]">
            <div className="grid h-full w-full lg:h-[min(82dvh,680px)] lg:grid-cols-[minmax(0,0.94fr)_minmax(320px,0.96fr)]">
            <section className="order-2 flex min-h-full bg-transparent px-6 py-8 sm:px-10 lg:order-1 lg:px-12 lg:py-10 xl:px-14 xl:py-12">
              <div className="mx-auto flex h-full w-full max-w-[360px] flex-col">
                <div className="shrink-0">
                  <button
                    type="button"
                    onClick={onBack}
                    className="inline-flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-white transition hover:text-white"
                  >
                    <i className="fa-solid fa-arrow-left text-[10px]" aria-hidden="true" />
                    Back to ReelAI
                  </button>
                </div>

                {account ? (
                  account.isVerified ? (
                    <div
                      className={`flex flex-1 flex-col justify-center pb-6 transition-opacity duration-150 lg:pb-0 ${
                        authContentVisible ? "opacity-100" : "opacity-0"
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
                          onClick={onOpenYourSets}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition hover:bg-[#2a2a2a]"
                        >
                          Open Your Sets
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setShowChangePassword((prev) => !prev);
                            setChangePasswordError(null);
                            setChangePasswordSuccess(null);
                            setCurrentPassword("");
                            setNewPassword("");
                          }}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition hover:bg-[#2a2a2a]"
                        >
                          {showChangePassword ? "Cancel" : "Change Password"}
                        </button>
                        {showChangePassword ? (
                          <form onSubmit={onChangePassword} className="mt-1 flex flex-col gap-3">
                            <input
                              type="password"
                              value={currentPassword}
                              onChange={(event) => setCurrentPassword(event.target.value)}
                              autoComplete="current-password"
                              placeholder="Current password"
                              className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 text-sm text-white outline-none placeholder:text-white placeholder:opacity-100 backdrop-blur-[10px]"
                            />
                            <input
                              type="password"
                              value={newPassword}
                              onChange={(event) => setNewPassword(event.target.value)}
                              autoComplete="new-password"
                              placeholder="New password (min 8 characters)"
                              className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 text-sm text-white outline-none placeholder:text-white placeholder:opacity-100 backdrop-blur-[10px]"
                            />
                            {changePasswordError ? <p className="text-sm text-[#ffb4b4]">{changePasswordError}</p> : null}
                            {changePasswordSuccess ? <p className="text-sm text-[#9ef8cb]">{changePasswordSuccess}</p> : null}
                            <button
                              type="submit"
                              disabled={changePasswordBusy}
                              className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition hover:bg-[#2a2a2a] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
                            >
                              {changePasswordBusy ? "Changing..." : "Update Password"}
                            </button>
                          </form>
                        ) : null}
                        <button
                          type="button"
                          onClick={onSignOut}
                          disabled={busy || changePasswordBusy}
                          className="inline-flex h-12 w-full items-center justify-center rounded-[14px] bg-[#1f1f1f] px-5 text-sm font-semibold text-white transition hover:bg-[#2a2a2a] disabled:cursor-not-allowed disabled:bg-[#1f1f1f]/45 disabled:text-white/35"
                        >
                          {busy ? "Signing out..." : "Sign Out"}
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div
                      className={`flex flex-1 flex-col justify-center pb-6 transition-opacity duration-150 lg:pb-0 ${
                        authContentVisible ? "opacity-100" : "opacity-0"
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
                          <input
                            type="email"
                            value={email}
                            onChange={(event) => setEmail(event.target.value)}
                            autoComplete="email"
                            placeholder="you@example.com"
                            className="community-auth-input h-12 w-full rounded-[12px] bg-[#404040] px-4 text-sm text-white outline-none placeholder:text-white placeholder:opacity-100 backdrop-blur-[10px]"
                          />
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
                        disabled={busy}
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
    </main>
  );
}
