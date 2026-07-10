"use client";

import { useEffect, useRef } from "react";

import { ViewportModalPortal } from "@/components/ViewportModalPortal";
import type { AssessmentSession } from "@/lib/types";

export type RecallAnswerReveal = {
  questionId: string;
  choiceIndex: number;
  correct: boolean;
  correctIndex: number;
  explanation: string;
};

type RecallCheckProps = {
  session: AssessmentSession;
  questionIndex: number;
  answerReveal: RecallAnswerReveal | null;
  answering: boolean;
  showResults: boolean;
  preparingFeed: boolean;
  snoozing: boolean;
  error: string | null;
  onAnswer: (choiceIndex: number) => void;
  onNextQuestion: () => void;
  onLater: () => void;
  onContinue: () => void;
};

function formatAccuracy(value: number | null | undefined): string | null {
  if (!Number.isFinite(value)) {
    return null;
  }
  return `${Math.round(Math.max(0, Math.min(1, Number(value))) * 100)}%`;
}

export function RecallCheck({
  session,
  questionIndex,
  answerReveal,
  answering,
  showResults,
  preparingFeed,
  snoozing,
  error,
  onAnswer,
  onNextQuestion,
  onLater,
  onContinue,
}: RecallCheckProps) {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const headingRef = useRef<HTMLHeadingElement | null>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const question = session.questions[questionIndex] ?? null;
  const recentAccuracy = formatAccuracy(session.recent_accuracy);
  const rollingAccuracy = formatAccuracy(session.rolling_accuracy);
  const correctCount = Math.round(
    Math.max(0, Math.min(1, Number(session.score) || 0)) * Math.max(0, session.question_count),
  );

  useEffect(() => {
    returnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    return () => returnFocusRef.current?.focus();
  }, []);

  useEffect(() => {
    headingRef.current?.focus();
  }, [question?.id, showResults]);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !showResults && !answering && !snoozing) {
        event.preventDefault();
        onLater();
        return;
      }
      if (event.key !== "Tab") {
        return;
      }
      const focusable = Array.from(
        dialog.querySelectorAll<HTMLElement>(
          "button:not(:disabled), input:not(:disabled), [href], [tabindex]:not([tabindex='-1'])",
        ),
      ).filter((element) => element.offsetParent !== null);
      if (focusable.length === 0) {
        event.preventDefault();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (!focusable.includes(document.activeElement as HTMLElement)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    dialog.addEventListener("keydown", onKeyDown);
    return () => dialog.removeEventListener("keydown", onKeyDown);
  }, [answering, onLater, showResults, snoozing]);

  return (
    <ViewportModalPortal>
      <div
        ref={dialogRef}
        className="fixed inset-0 z-[2147483647] overflow-y-auto bg-black/88 px-4 py-5 text-white backdrop-blur-2xl sm:px-6 sm:py-8"
        role="dialog"
        aria-modal="true"
        aria-labelledby="recall-check-title"
      >
        <div aria-hidden="true" className="pointer-events-none fixed inset-0 overflow-hidden">
          <div className="absolute left-1/2 top-[-25rem] h-[48rem] w-[48rem] -translate-x-1/2 rounded-full border border-white/10 bg-[radial-gradient(circle,rgba(255,255,255,0.12)_0%,rgba(255,255,255,0.025)_36%,transparent_68%)]" />
          <div className="absolute inset-x-0 top-1/2 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
        </div>

        <div className="relative mx-auto flex min-h-full w-full max-w-4xl items-center justify-center">
          <section className="w-full overflow-hidden rounded-[2rem] border border-white/16 bg-black/64 shadow-[0_30px_100px_rgba(0,0,0,0.7)] backdrop-blur-3xl">
            <header className="flex items-center justify-between gap-4 border-b border-white/10 px-5 py-4 sm:px-8 sm:py-5">
              <div className="flex items-center gap-3">
                <span className="grid h-9 w-9 place-items-center rounded-full border border-white/20 bg-white/[0.06] text-xs font-semibold">
                  <i className="fa-solid fa-rotate-left" aria-hidden="true" />
                </span>
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-white/48">Adaptive learning</p>
                  <p className="mt-0.5 text-sm font-semibold text-white/92">Recall Check</p>
                </div>
              </div>
              {!showResults ? (
                <button
                  type="button"
                  onClick={onLater}
                  disabled={answering || snoozing}
                  className="rounded-full border border-white/16 px-4 py-2 text-xs font-semibold text-white/68 transition hover:border-white/32 hover:text-white disabled:cursor-wait disabled:opacity-45"
                >
                  {snoozing ? "Saving..." : "Later"}
                </button>
              ) : null}
            </header>

            {showResults ? (
              <div className="px-5 py-8 sm:px-10 sm:py-10">
                <div className="grid gap-8 lg:grid-cols-[0.8fr_1.2fr] lg:items-start">
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-white/45">Check complete</p>
                    <h2
                      ref={headingRef}
                      id="recall-check-title"
                      tabIndex={-1}
                      className="mt-3 text-4xl font-semibold tracking-[-0.04em] outline-none sm:text-5xl"
                    >
                      {correctCount} of {session.question_count}
                    </h2>
                    <p className="mt-3 max-w-sm text-sm leading-6 text-white/58">
                      Your next reels are being tuned around what held and what needs another pass.
                    </p>
                    <div className="mt-6 flex flex-wrap gap-2">
                      {recentAccuracy ? (
                        <span className="rounded-full border border-white/14 bg-white/[0.05] px-3 py-1.5 text-xs text-white/72">
                          Recent {recentAccuracy}
                        </span>
                      ) : null}
                      {rollingAccuracy ? (
                        <span className="rounded-full border border-white/14 bg-white/[0.05] px-3 py-1.5 text-xs text-white/72">
                          Rolling {rollingAccuracy}
                        </span>
                      ) : null}
                    </div>
                  </div>

                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="rounded-3xl border border-white/12 bg-white/[0.045] p-5">
                      <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-white/48">Understood</p>
                      {session.understood_concepts.length > 0 ? (
                        <ul className="mt-3 space-y-2 text-sm text-white/88">
                          {session.understood_concepts.map((concept) => (
                            <li key={`understood-${concept}`} className="flex items-start gap-2">
                              <i className="fa-solid fa-check mt-1 text-[10px] text-white/60" aria-hidden="true" />
                              <span>{concept}</span>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="mt-3 text-sm leading-6 text-white/48">Keep building the signal with the next set.</p>
                      )}
                    </div>
                    <div className="rounded-3xl border border-white/12 bg-white/[0.045] p-5">
                      <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-white/48">Revisit</p>
                      {session.revisit_concepts.length > 0 ? (
                        <ul className="mt-3 space-y-2 text-sm text-white/88">
                          {session.revisit_concepts.map((concept) => (
                            <li key={`revisit-${concept}`} className="flex items-start gap-2">
                              <i className="fa-solid fa-arrow-rotate-right mt-1 text-[10px] text-white/60" aria-hidden="true" />
                              <span>{concept}</span>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="mt-3 text-sm leading-6 text-white/48">Nothing urgent surfaced in this check.</p>
                      )}
                    </div>
                  </div>
                </div>

                {error ? <p className="mt-6 text-sm text-white/72" role="alert">{error}</p> : null}
                <div className="mt-8 flex justify-end">
                  <button
                    type="button"
                    onClick={onContinue}
                    disabled={preparingFeed}
                    className="inline-flex min-w-44 items-center justify-center gap-2 rounded-full bg-white px-6 py-3 text-sm font-semibold text-black transition hover:bg-white/88 disabled:cursor-wait disabled:bg-white/55"
                  >
                    {preparingFeed ? "Tuning your feed..." : "Continue learning"}
                    {!preparingFeed ? <i className="fa-solid fa-arrow-right text-xs" aria-hidden="true" /> : null}
                  </button>
                </div>
              </div>
            ) : question ? (
              <div className="px-5 py-7 sm:px-10 sm:py-9">
                <div className="flex items-center gap-4">
                  <p className="shrink-0 text-[10px] font-semibold uppercase tracking-[0.16em] text-white/45">
                    Question {questionIndex + 1} / {session.question_count}
                  </p>
                  <div className="h-px flex-1 overflow-hidden bg-white/10">
                    <div
                      className="h-full bg-white/70 transition-[width] duration-300"
                      style={{ width: `${Math.min(100, ((questionIndex + 1) / Math.max(1, session.question_count)) * 100)}%` }}
                    />
                  </div>
                </div>

                <p className="mt-6 text-[10px] font-semibold uppercase tracking-[0.16em] text-white/42">
                  {question.concept_title}
                </p>
                <h2
                  ref={headingRef}
                  id="recall-check-title"
                  tabIndex={-1}
                  className="mt-2 max-w-3xl text-2xl font-semibold leading-tight tracking-[-0.025em] outline-none sm:text-3xl"
                >
                  {question.prompt}
                </h2>

                <div className="mt-7 grid gap-3 sm:grid-cols-2" role="group" aria-label="Answer choices">
                  {question.options.slice(0, 4).map((option, index) => {
                    const selected = answerReveal?.choiceIndex === index;
                    const correct = answerReveal?.correctIndex === index;
                    const revealedWrong = Boolean(answerReveal) && selected && !correct;
                    const stateClass = correct
                      ? "border-white/72 bg-white text-black"
                      : revealedWrong
                        ? "border-white/40 bg-white/[0.13] text-white/64"
                        : selected
                          ? "border-white/48 bg-white/[0.1] text-white"
                          : "border-white/14 bg-white/[0.035] text-white/86 hover:border-white/32 hover:bg-white/[0.07]";
                    return (
                      <button
                        key={`${question.id}-option-${index}`}
                        type="button"
                        onClick={() => onAnswer(index)}
                        disabled={answering || Boolean(answerReveal)}
                        aria-pressed={selected}
                        className={`group flex min-h-20 items-start gap-3 rounded-2xl border px-4 py-4 text-left text-sm leading-5 transition disabled:cursor-default ${stateClass}`}
                      >
                        <span className={`grid h-7 w-7 shrink-0 place-items-center rounded-full border text-[10px] font-semibold ${correct ? "border-black/20" : "border-current/20"}`}>
                          {String.fromCharCode(65 + index)}
                        </span>
                        <span className="pt-0.5">{option}</span>
                      </button>
                    );
                  })}
                </div>

                {answerReveal ? (
                  <div className="mt-6 rounded-2xl border border-white/12 bg-white/[0.05] p-4" role="status" aria-live="polite">
                    <p className="text-xs font-semibold text-white">
                      {answerReveal.correct ? "That’s right." : "Not quite — here’s the grounded answer."}
                    </p>
                    <p className="mt-2 text-sm leading-6 text-white/66">{answerReveal.explanation}</p>
                  </div>
                ) : null}
                {error ? <p className="mt-4 text-sm text-white/72" role="alert">{error}</p> : null}

                <div className="mt-7 flex justify-end">
                  {answerReveal ? (
                    <button
                      type="button"
                      onClick={onNextQuestion}
                      className="inline-flex min-w-40 items-center justify-center gap-2 rounded-full bg-white px-5 py-2.5 text-sm font-semibold text-black transition hover:bg-white/88"
                    >
                      {session.answered_count >= session.question_count ? "See results" : "Next question"}
                      <i className="fa-solid fa-arrow-right text-xs" aria-hidden="true" />
                    </button>
                  ) : null}
                </div>
              </div>
            ) : (
              <div className="px-8 py-12 text-center">
                <h2 ref={headingRef} id="recall-check-title" tabIndex={-1} className="text-xl font-semibold outline-none">
                  Preparing your recall check
                </h2>
              </div>
            )}
          </section>
        </div>
      </div>
    </ViewportModalPortal>
  );
}
