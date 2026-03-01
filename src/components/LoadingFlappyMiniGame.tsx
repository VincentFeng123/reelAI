"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const WIDTH = 320;
const HEIGHT = 156;
const FLOOR_HEIGHT = 10;
const BIRD_X = 78;
const BIRD_RADIUS = 6;
const PIPE_WIDTH = 24;
const PIPE_GAP = 44;
const PIPE_SPEED = 1.9;
const GRAVITY = 0.24;
const FLAP_VELOCITY = -4.3;
const BEST_SCORE_KEY = "studyreels-flappy-best";

type Pipe = {
  x: number;
  gapY: number;
  passed: boolean;
};

type GameState = {
  started: boolean;
  alive: boolean;
  score: number;
  best: number;
  birdY: number;
  birdVY: number;
  pipes: Pipe[];
  spawnCooldown: number;
};

function createInitialState(best: number): GameState {
  return {
    started: false,
    alive: true,
    score: 0,
    best,
    birdY: HEIGHT * 0.5,
    birdVY: 0,
    pipes: [],
    spawnCooldown: 46,
  };
}

function nextPipeGapY(): number {
  const min = 26;
  const max = HEIGHT - FLOOR_HEIGHT - 26 - PIPE_GAP;
  return Math.floor(min + Math.random() * Math.max(1, max - min));
}

export function LoadingFlappyMiniGame() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameRef = useRef<number | null>(null);
  const stateRef = useRef<GameState>(createInitialState(0));

  const [started, setStarted] = useState(false);
  const [alive, setAlive] = useState(true);
  const [score, setScore] = useState(0);
  const [best, setBest] = useState(0);

  const syncUi = useCallback((state: GameState) => {
    setStarted(state.started);
    setAlive(state.alive);
    setScore(state.score);
    setBest(state.best);
  }, []);

  const persistBest = useCallback((nextBest: number) => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(BEST_SCORE_KEY, String(nextBest));
  }, []);

  const resetRound = useCallback((state: GameState, keepStarted = false) => {
    state.started = keepStarted;
    state.alive = true;
    state.score = 0;
    state.birdY = HEIGHT * 0.5;
    state.birdVY = 0;
    state.pipes = [];
    state.spawnCooldown = 46;
  }, []);

  const flap = useCallback(() => {
    const state = stateRef.current;
    if (!state.started) {
      resetRound(state, true);
      state.birdVY = FLAP_VELOCITY;
      syncUi(state);
      return;
    }
    if (!state.alive) {
      resetRound(state, true);
      state.birdVY = FLAP_VELOCITY;
      syncUi(state);
      return;
    }
    state.birdVY = FLAP_VELOCITY;
  }, [resetRound, syncUi]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const bestRaw = Number(window.localStorage.getItem(BEST_SCORE_KEY));
    const safeBest = Number.isFinite(bestRaw) && bestRaw > 0 ? Math.floor(bestRaw) : 0;
    stateRef.current.best = safeBest;
    setBest(safeBest);
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) {
        return;
      }
      if (event.code === "Space" || event.code === "ArrowUp") {
        event.preventDefault();
        flap();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [flap]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    const state = stateRef.current;
    const draw = () => {
      ctx.clearRect(0, 0, WIDTH, HEIGHT);

      ctx.fillStyle = "#050505";
      ctx.fillRect(0, 0, WIDTH, HEIGHT);

      ctx.strokeStyle = "rgba(255,255,255,0.18)";
      ctx.lineWidth = 1;
      for (let x = 0; x <= WIDTH; x += 24) {
        ctx.beginPath();
        ctx.moveTo(x + 0.5, 0);
        ctx.lineTo(x + 0.5, HEIGHT);
        ctx.stroke();
      }
      for (let y = 0; y <= HEIGHT; y += 24) {
        ctx.beginPath();
        ctx.moveTo(0, y + 0.5);
        ctx.lineTo(WIDTH, y + 0.5);
        ctx.stroke();
      }

      ctx.fillStyle = "rgba(255,255,255,0.9)";
      for (const pipe of state.pipes) {
        const topHeight = pipe.gapY;
        const bottomY = pipe.gapY + PIPE_GAP;
        const bottomHeight = HEIGHT - FLOOR_HEIGHT - bottomY;
        ctx.fillRect(pipe.x, 0, PIPE_WIDTH, topHeight);
        ctx.fillRect(pipe.x, bottomY, PIPE_WIDTH, bottomHeight);
      }

      ctx.fillStyle = "#fff";
      ctx.beginPath();
      ctx.arc(BIRD_X, state.birdY, BIRD_RADIUS, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#050505";
      ctx.beginPath();
      ctx.arc(BIRD_X + 2, state.birdY - 1, 1.4, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#fff";
      ctx.fillRect(0, HEIGHT - FLOOR_HEIGHT, WIDTH, FLOOR_HEIGHT);
    };

    const tick = () => {
      const current = stateRef.current;
      if (current.started && current.alive) {
        current.birdVY += GRAVITY;
        current.birdY += current.birdVY;

        current.spawnCooldown -= 1;
        if (current.spawnCooldown <= 0) {
          current.spawnCooldown = 74;
          current.pipes.push({
            x: WIDTH + 8,
            gapY: nextPipeGapY(),
            passed: false,
          });
        }

        for (const pipe of current.pipes) {
          pipe.x -= PIPE_SPEED;

          const birdFront = BIRD_X + BIRD_RADIUS;
          const birdBack = BIRD_X - BIRD_RADIUS;
          const inPipeX = birdFront >= pipe.x && birdBack <= pipe.x + PIPE_WIDTH;
          const outsideGap = current.birdY - BIRD_RADIUS < pipe.gapY || current.birdY + BIRD_RADIUS > pipe.gapY + PIPE_GAP;
          if (inPipeX && outsideGap) {
            current.alive = false;
          }

          if (!pipe.passed && pipe.x + PIPE_WIDTH < BIRD_X - BIRD_RADIUS) {
            pipe.passed = true;
            current.score += 1;
            setScore(current.score);
            if (current.score > current.best) {
              current.best = current.score;
              setBest(current.best);
              persistBest(current.best);
            }
          }
        }

        current.pipes = current.pipes.filter((pipe) => pipe.x + PIPE_WIDTH > -2);

        if (current.birdY + BIRD_RADIUS >= HEIGHT - FLOOR_HEIGHT || current.birdY - BIRD_RADIUS <= 0) {
          current.alive = false;
        }

        if (!current.alive) {
          syncUi(current);
        }
      }

      draw();
      frameRef.current = window.requestAnimationFrame(tick);
    };

    frameRef.current = window.requestAnimationFrame(tick);
    return () => {
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
    };
  }, [persistBest, syncUi]);

  return (
    <div className="mx-auto mt-3 w-full max-w-[360px] rounded-2xl border border-white/20 bg-black/75 p-2.5 text-white backdrop-blur-sm">
      <div className="mb-2 flex items-center justify-between text-[10px] font-semibold uppercase tracking-[0.14em] text-white/70">
        <span>Flappy Break</span>
        <span>
          {score} / {best}
        </span>
      </div>
      <button
        type="button"
        onPointerDown={(event) => {
          event.preventDefault();
          flap();
        }}
        className="relative block w-full overflow-hidden rounded-xl border border-white/18 bg-black/90"
        aria-label="Play flappy mini game"
      >
        <canvas ref={canvasRef} width={WIDTH} height={HEIGHT} className="block h-[156px] w-full" />
        {!started ? (
          <div className="pointer-events-none absolute inset-0 grid place-items-center bg-black/30">
            <p className="rounded-full border border-white/20 bg-black/65 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-white/82">
              Tap or Space
            </p>
          </div>
        ) : null}
        {started && !alive ? (
          <div className="pointer-events-none absolute inset-0 grid place-items-center bg-black/40">
            <p className="rounded-full border border-white/25 bg-black/70 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-white">
              Crashed, tap to retry
            </p>
          </div>
        ) : null}
      </button>
    </div>
  );
}
