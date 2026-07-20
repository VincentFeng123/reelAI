"use client";

import { useEffect, useRef } from "react";

import type { AssessmentSession } from "@/lib/types";

export type RecallAnswerReveal = {
  questionId: string;
  choiceIndex: number;
  correct: boolean;
  correctIndex: number;
  explanation: string;
};

type RecallCheckProps = {
  active: boolean;
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

const QUIET_ACTION_CLASS =
  "rounded-xl bg-transparent px-4 py-2 text-sm font-semibold text-white/68 transition-colors enabled:hover:bg-white/[0.07] enabled:hover:text-white focus-visible:bg-white/[0.07] focus-visible:text-white focus-visible:outline-none disabled:cursor-default disabled:opacity-40";

export function RecallCheck({
  active,
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
  const headingRef = useRef<HTMLHeadingElement | null>(null);
  const question = session.questions[questionIndex] ?? null;
  const recentAccuracy = formatAccuracy(session.recent_accuracy);
  const rollingAccuracy = formatAccuracy(session.rolling_accuracy);
  const correctCount = Math.round(
    Math.max(0, Math.min(1, Number(session.score) || 0)) * Math.max(0, session.question_count),
  );

  useEffect(() => {
    if (active) {
      headingRef.current?.focus({ preventScroll: true });
    }
  }, [active, question?.id, showResults]);

  return (
    <section
      data-feed-item="recall-check"
      aria-hidden={!active}
      inert={!active}
      role="region"
      aria-labelledby="recall-check-title"
      className="h-full w-full overflow-y-auto overscroll-contain bg-transparent text-white"
    >
      <div className="mx-auto flex min-h-full w-full max-w-[29.25rem] flex-col px-5 py-10 sm:px-6 sm:py-12">
        <header className="flex items-start justify-between gap-4">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-white/38">Adaptive learning</p>
            <p className="mt-1 text-sm font-semibold text-white/82">Recall Check</p>
            {question ? <p className="mt-1 text-xs text-white/42">{question.concept_title}</p> : null}
          </div>
          {!showResults ? (
            <button
              type="button"
              data-reel-control="true"
              onClick={onLater}
              disabled={answering || snoozing}
              className={QUIET_ACTION_CLASS}
            >
              {snoozing ? "Saving..." : "Later"}
            </button>
          ) : null}
        </header>

        {showResults ? (
          <div className="flex flex-1 items-center py-8">
            <div className="w-full">
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-white/38">Check complete</p>
                <h2
                  ref={headingRef}
                  id="recall-check-title"
                  tabIndex={-1}
                  className="mt-3 text-4xl font-semibold tracking-[-0.05em] outline-none sm:text-5xl"
                >
                  {correctCount}/{session.question_count}
                </h2>
                <p className="mt-4 text-sm leading-6 text-white/52">
                  Your next reels will lean into what held and revisit what needs another pass.
                </p>
                {recentAccuracy || rollingAccuracy ? (
                  <div className="mt-5 flex flex-wrap gap-x-6 gap-y-2 text-xs text-white/48">
                    {recentAccuracy ? <span>Recent recall {recentAccuracy}</span> : null}
                    {rollingAccuracy ? <span>Rolling recall {rollingAccuracy}</span> : null}
                  </div>
                ) : null}
              </div>

              <div className="mt-8">
                <div className="grid gap-7 sm:grid-cols-2 sm:gap-8">
                  <section>
                    <h3 className="text-[10px] font-semibold uppercase tracking-[0.18em] text-white/42">Understood</h3>
                    {session.understood_concepts.length > 0 ? (
                      <ul className="mt-4 space-y-3 text-sm text-white/82">
                        {session.understood_concepts.map((concept) => (
                          <li key={`understood-${concept}`} className="flex items-start gap-3">
                            <span aria-hidden="true" className="mt-[0.48rem] h-1 w-1 shrink-0 rounded-full bg-white/45" />
                            <span>{concept}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="mt-4 text-sm leading-6 text-white/42">Keep building the signal with the next set.</p>
                    )}
                  </section>
                  <section>
                    <h3 className="text-[10px] font-semibold uppercase tracking-[0.18em] text-white/42">Revisit</h3>
                    {session.revisit_concepts.length > 0 ? (
                      <ul className="mt-4 space-y-3 text-sm text-white/82">
                        {session.revisit_concepts.map((concept) => (
                          <li key={`revisit-${concept}`} className="flex items-start gap-3">
                            <span aria-hidden="true" className="mt-[0.48rem] h-1 w-1 shrink-0 rounded-full bg-white/25" />
                            <span>{concept}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="mt-4 text-sm leading-6 text-white/42">Nothing urgent surfaced in this check.</p>
                    )}
                  </section>
                </div>

                {error ? <p className="mt-6 text-sm text-white/64" role="alert">{error}</p> : null}
                <div className="mt-8 flex justify-end">
                  <button
                    type="button"
                    data-reel-control="true"
                    onClick={onContinue}
                    disabled={preparingFeed}
                    className={QUIET_ACTION_CLASS}
                  >
                    {preparingFeed ? "Tuning your feed..." : "Continue learning"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : question ? (
          <div className="flex flex-1 flex-col justify-center py-7 sm:py-8">
            <div
              role="progressbar"
              aria-label="Quiz progress"
              aria-valuemin={1}
              aria-valuemax={Math.max(1, session.question_count)}
              aria-valuenow={Math.min(questionIndex + 1, Math.max(1, session.question_count))}
              className="mb-3 h-0.5 w-full overflow-hidden rounded-full bg-white/[0.08]"
            >
              <div
                className="h-full rounded-full bg-white/60 transition-[width] duration-300"
                style={{ width: `${Math.min(100, ((questionIndex + 1) / Math.max(1, session.question_count)) * 100)}%` }}
              />
            </div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-white/38">
              Question {questionIndex + 1} out of {session.question_count}
            </p>
            <h2
              ref={headingRef}
              id="recall-check-title"
              tabIndex={-1}
              className="mt-3 text-xl font-semibold leading-[1.14] tracking-[-0.025em] outline-none sm:text-2xl"
            >
              {question.prompt}
            </h2>

            <div className="mt-6 grid grid-cols-1 gap-2" role="group" aria-label="Answer choices">
              {question.options.slice(0, 4).map((option, index) => {
                const selected = answerReveal?.choiceIndex === index;
                const correct = answerReveal?.correctIndex === index;
                const stateClass = answerReveal
                  ? correct
                    ? "bg-emerald-500/20 text-emerald-100"
                    : selected
                      ? "bg-red-500/20 text-red-100"
                      : "bg-white/[0.035] text-white/32"
                  : "bg-white/[0.045] text-white/78 enabled:hover:bg-white/[0.07] enabled:hover:text-white";
                const outcome = answerReveal
                  ? correct
                    ? selected
                      ? "Correct"
                      : "Correct answer"
                    : selected
                      ? "Your answer"
                      : null
                  : null;
                return (
                  <button
                    key={`${question.id}-option-${index}`}
                    type="button"
                    data-reel-control="true"
                    data-recall-choice="true"
                    onClick={() => onAnswer(index)}
                    disabled={answering || Boolean(answerReveal)}
                    aria-pressed={selected}
                    className={`group flex min-h-[3.75rem] items-center gap-3 rounded-xl px-4 py-3 text-left transition-colors duration-150 focus-visible:bg-white/[0.07] focus-visible:outline-none disabled:cursor-default ${stateClass}`}
                  >
                    <span className="w-5 shrink-0 text-[10px] font-semibold uppercase tracking-[0.16em] text-current/55">
                      {String.fromCharCode(65 + index)}
                    </span>
                    <span className="min-w-0 flex-1 text-sm font-medium leading-5">{option}</span>
                    {outcome ? (
                      <span className="shrink-0 text-[9px] font-semibold uppercase tracking-[0.14em] text-current/58">{outcome}</span>
                    ) : null}
                  </button>
                );
              })}
            </div>

            {answerReveal ? (
              <div className="mt-5" role="status" aria-live="polite">
                <p className="text-xs font-semibold text-white/88">
                  {answerReveal.correct ? "That’s right." : "Not quite. Here’s the grounded answer."}
                </p>
                <p className="mt-2 text-sm leading-6 text-white/56">{answerReveal.explanation}</p>
              </div>
            ) : null}
            {error ? <p className="mt-5 text-sm text-white/64" role="alert">{error}</p> : null}

            {answerReveal ? (
              <div className="mt-5 flex justify-end">
                <button
                  type="button"
                  data-reel-control="true"
                  onClick={onNextQuestion}
                  className={QUIET_ACTION_CLASS}
                >
                  {session.answered_count >= session.question_count ? "See results" : "Next question"}
                </button>
              </div>
            ) : null}
          </div>
        ) : (
          <div className="flex flex-1 items-center justify-center py-12 text-center">
            <div>
              <h2 ref={headingRef} id="recall-check-title" tabIndex={-1} className="text-2xl font-semibold tracking-tight outline-none">
                Preparing your recall check
              </h2>
              <p className="mt-3 text-sm text-white/48">Choosing a few grounded questions from what you just watched.</p>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
