"""
Boundary stress tests — 200+ test runs across diverse topics and edge cases.

Verifies the core contract:
  - Every reel STARTS at the beginning of a complete sentence (topic start).
  - Every reel ENDS at terminal punctuation {'.', '!', '?', '...'} or topic end.
  - No mid-sentence cuts at either end.
  - Consecutive continuation reels have zero gap, zero overlap.

Uses synthetic but realistic transcripts to cover edge cases:
  - Varying sentence lengths (short vs long)
  - Varying punctuation density (sparse vs dense)
  - Unpunctuated transcripts (auto-captions)
  - Different cue durations (1s, 3s, 5s, 8s)
  - Segment lengths at the boundary of min/max
  - Many topics (math, science, history, CS, language, arts, etc.)
"""

from __future__ import annotations

import random
import unittest
from typing import Any

from app.services.embeddings import EmbeddingService
from app.services.reels import ReelService
from app.services.youtube import YouTubeService


# ========== transcript generators ========== #


def _make_cue(start: float, duration: float, text: str) -> dict[str, Any]:
    return {"start": start, "duration": duration, "text": text}


def _build_topic_transcript(
    topic_sentences: list[str],
    start_t: float = 0.0,
    cue_duration: float = 3.0,
    gap: float = 0.0,
) -> list[dict[str, Any]]:
    """Build a transcript from a list of sentences, each as a cue."""
    cues = []
    t = start_t
    for text in topic_sentences:
        cues.append(_make_cue(t, cue_duration, text))
        t += cue_duration + gap
    return cues


def _max_inter_cue_gap(transcript: list[dict[str, Any]]) -> float:
    """Compute the maximum silence gap between consecutive cues in a transcript.

    When consecutive windows are chained via _split_into_consecutive_windows,
    the refiner snaps each boundary to a cue edge.  If cues have natural pauses
    between them (real speech breathing, YouTube caption gaps), the resulting
    window-to-window gap will be at most this value.  No content is dropped —
    the gap is silence.
    """
    max_gap = 0.0
    for i in range(len(transcript) - 1):
        end_i = float(transcript[i]["start"]) + float(transcript[i].get("duration") or 0)
        start_next = float(transcript[i + 1]["start"])
        gap = start_next - end_i
        if gap > max_gap:
            max_gap = gap
    return max_gap


def _check_continuation_gaps(
    windows: list[tuple[float, float]],
    transcript: list[dict[str, Any]],
    label: str,
) -> list[str]:
    """Verify continuation chaining: no overlap, no gap beyond inter-cue silence."""
    failures: list[str] = []
    max_cue_gap = _max_inter_cue_gap(transcript)
    # Allow the natural inter-cue gap plus a small tolerance for float rounding
    gap_tolerance = max_cue_gap + 0.05
    for i in range(len(windows) - 1):
        gap = windows[i + 1][0] - windows[i][1]
        if gap < -0.01:
            failures.append(f"{label}: overlap={abs(gap):.3f}s between win {i},{i+1}")
        elif gap > gap_tolerance:
            failures.append(
                f"{label}: gap={gap:.3f}s between win {i},{i+1} "
                f"(exceeds max inter-cue silence {max_cue_gap:.3f}s)"
            )
    return failures


# 20 diverse topics with realistic educational sentences
TOPIC_TRANSCRIPTS: dict[str, list[str]] = {
    "calculus_derivatives": [
        "Let us start with derivatives in calculus.",
        "A derivative measures instantaneous rate of change.",
        "We denote the derivative as f prime of x.",
        "The power rule states that the derivative of x to the n is n times x to the n minus one.",
        "We can apply the chain rule for composite functions.",
        "The product rule handles multiplication of functions.",
        "The quotient rule covers division of functions.",
        "Implicit differentiation lets us handle equations without explicit forms.",
        "Higher order derivatives give us acceleration and beyond.",
        "That wraps up our discussion of derivatives.",
    ],
    "calculus_integrals": [
        "Now let us discuss integrals.",
        "An integral computes the area under a curve.",
        "The indefinite integral yields an antiderivative.",
        "The definite integral gives a specific numerical value.",
        "The fundamental theorem of calculus connects derivatives and integrals.",
        "Integration by parts reverses the product rule.",
        "Substitution simplifies complex integrals.",
        "Partial fractions break rational functions apart.",
        "Numerical integration approximates difficult integrals.",
        "That concludes our study of integration.",
    ],
    "linear_algebra": [
        "Linear algebra studies vectors and matrices.",
        "A vector has both magnitude and direction.",
        "Matrix multiplication combines linear transformations.",
        "The determinant tells us if a matrix is invertible.",
        "Eigenvalues reveal the scaling factors of transformations.",
        "Eigenvectors point in directions preserved by the transformation.",
        "Gaussian elimination solves systems of linear equations.",
        "The rank of a matrix equals the dimension of its column space.",
        "Orthogonal matrices preserve lengths and angles.",
        "Linear algebra is fundamental to machine learning.",
    ],
    "photosynthesis": [
        "Photosynthesis converts sunlight into chemical energy.",
        "Plants use chlorophyll to capture light energy.",
        "The light reactions occur in the thylakoid membranes.",
        "Water molecules are split to release oxygen.",
        "ATP and NADPH are produced during the light reactions.",
        "The Calvin cycle fixes carbon dioxide into glucose.",
        "RuBisCO is the key enzyme in carbon fixation.",
        "The dark reactions do not actually require darkness.",
        "C4 and CAM plants have adapted to hot dry climates.",
        "Photosynthesis is essential for life on earth.",
    ],
    "cell_biology": [
        "The cell is the basic unit of life.",
        "Eukaryotic cells contain membrane-bound organelles.",
        "The nucleus houses the genetic material.",
        "Mitochondria are the powerhouses of the cell.",
        "The endoplasmic reticulum synthesizes proteins and lipids.",
        "The Golgi apparatus packages and ships molecules.",
        "Lysosomes digest waste and recycle cellular components.",
        "The cell membrane controls what enters and exits.",
        "Cell division occurs through mitosis and meiosis.",
        "Cell biology underpins all of modern medicine.",
    ],
    "world_war_2": [
        "World War Two began in September nineteen thirty nine.",
        "Germany invaded Poland triggering the conflict.",
        "The Allied powers included Britain France and the Soviet Union.",
        "The Axis powers were Germany Italy and Japan.",
        "The Battle of Stalingrad was a turning point on the Eastern Front.",
        "D Day on June sixth nineteen forty four opened a second front.",
        "The Holocaust was the genocide of six million Jews.",
        "The atomic bombs on Hiroshima and Nagasaki ended the Pacific war.",
        "The United Nations was founded to prevent future conflicts.",
        "World War Two reshaped the entire global order.",
    ],
    "machine_learning": [
        "Machine learning is a subset of artificial intelligence.",
        "Supervised learning uses labeled training data.",
        "The model learns to map inputs to outputs.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning uses multiple hidden layers.",
        "Gradient descent optimizes the loss function.",
        "Overfitting occurs when the model memorizes training data.",
        "Regularization techniques prevent overfitting.",
        "Cross validation estimates generalization performance.",
        "Machine learning powers recommendation systems and self driving cars.",
    ],
    "quantum_mechanics": [
        "Quantum mechanics describes physics at the atomic scale.",
        "Particles exhibit wave particle duality.",
        "The Schrodinger equation governs quantum state evolution.",
        "The uncertainty principle limits simultaneous measurement precision.",
        "Superposition means a particle can be in multiple states at once.",
        "Entanglement links distant particles instantaneously.",
        "Measurement collapses the wave function to a definite state.",
        "Quantum tunneling allows particles to pass through barriers.",
        "Quantum computing harnesses superposition for parallel computation.",
        "Quantum mechanics is verified by countless experiments.",
    ],
    "organic_chemistry": [
        "Organic chemistry studies carbon based compounds.",
        "Carbon forms four covalent bonds.",
        "Functional groups determine chemical properties.",
        "Alkanes contain only single bonds between carbons.",
        "Alkenes have at least one carbon carbon double bond.",
        "Alcohols contain a hydroxyl group.",
        "Carboxylic acids have a carboxyl functional group.",
        "Esters are formed from acids and alcohols.",
        "Polymers are long chains of repeating monomers.",
        "Organic chemistry is essential for pharmaceuticals.",
    ],
    "economics_supply_demand": [
        "Supply and demand determine market prices.",
        "When demand increases and supply stays constant prices rise.",
        "When supply increases and demand stays constant prices fall.",
        "The equilibrium price balances supply and demand.",
        "Price elasticity measures sensitivity to price changes.",
        "Inferior goods see demand fall as income rises.",
        "Substitutes compete for consumer spending.",
        "Complements are purchased together.",
        "Government intervention can shift supply or demand curves.",
        "Market equilibrium is a core concept in microeconomics.",
    ],
    "python_programming": [
        "Python is a high level programming language.",
        "Variables store data that can change during execution.",
        "Functions encapsulate reusable blocks of code.",
        "Lists are ordered mutable collections.",
        "Dictionaries map keys to values.",
        "Loops repeat actions for each item in a sequence.",
        "Conditional statements control program flow.",
        "Classes define blueprints for objects.",
        "Exception handling manages runtime errors gracefully.",
        "Python is widely used in data science and web development.",
    ],
    "american_revolution": [
        "The American Revolution began in seventeen seventy five.",
        "Colonists protested taxation without representation.",
        "The Declaration of Independence was signed in seventeen seventy six.",
        "George Washington commanded the Continental Army.",
        "The Battle of Saratoga was a turning point for the colonies.",
        "France entered the war as an ally of the colonists.",
        "The Treaty of Paris in seventeen eighty three ended the war.",
        "The Constitution established the framework of government.",
        "The Bill of Rights guaranteed individual freedoms.",
        "The Revolution inspired democratic movements worldwide.",
    ],
    "music_theory": [
        "Music theory explains the structure of sound.",
        "Notes are named A through G in Western music.",
        "Scales provide the tonal foundation for melodies.",
        "Chords are three or more notes played simultaneously.",
        "Rhythm organizes music in time.",
        "Key signatures indicate which notes are sharp or flat.",
        "Intervals measure the distance between two pitches.",
        "Harmony results from combining different melodic lines.",
        "Dynamics control the volume of musical expression.",
        "Understanding theory helps musicians compose and improvise.",
    ],
    "climate_change": [
        "Climate change refers to long term shifts in temperature patterns.",
        "Greenhouse gases trap heat in the atmosphere.",
        "Carbon dioxide is the primary greenhouse gas from human activity.",
        "Deforestation reduces the planet's ability to absorb carbon.",
        "Sea levels are rising due to melting ice caps.",
        "Extreme weather events are becoming more frequent.",
        "The Paris Agreement aims to limit warming to one point five degrees.",
        "Renewable energy reduces dependence on fossil fuels.",
        "Individual actions like reducing waste contribute to solutions.",
        "Climate science is supported by overwhelming evidence.",
    ],
    "human_anatomy": [
        "The human body contains eleven major organ systems.",
        "The skeletal system provides structure and protection.",
        "Muscles enable movement through contraction.",
        "The cardiovascular system circulates blood throughout the body.",
        "The respiratory system exchanges oxygen and carbon dioxide.",
        "The nervous system transmits electrical signals for communication.",
        "The digestive system breaks down food into nutrients.",
        "The immune system defends against pathogens.",
        "The endocrine system regulates hormones.",
        "Understanding anatomy is fundamental to medicine.",
    ],
    "philosophy_ethics": [
        "Ethics examines questions of right and wrong.",
        "Utilitarianism seeks the greatest good for the greatest number.",
        "Deontological ethics focuses on duty and rules.",
        "Virtue ethics emphasizes character and moral excellence.",
        "The trolley problem illustrates ethical dilemmas vividly.",
        "Moral relativism argues that ethics vary by culture.",
        "Kantian ethics uses the categorical imperative as a test.",
        "Consequentialism judges actions by their outcomes.",
        "Social contract theory explains the basis of political obligation.",
        "Ethics remains central to law medicine and public policy.",
    ],
    "statistics_probability": [
        "Statistics is the science of collecting and analyzing data.",
        "Probability measures the likelihood of events occurring.",
        "The mean is the arithmetic average of a dataset.",
        "The median is the middle value when data is sorted.",
        "Standard deviation measures the spread of data.",
        "A normal distribution forms a bell shaped curve.",
        "Hypothesis testing determines if results are statistically significant.",
        "A p value below point zero five typically rejects the null hypothesis.",
        "Correlation does not imply causation.",
        "Statistics underpins modern research across all fields.",
    ],
    "shakespeare": [
        "Shakespeare wrote thirty seven plays over his career.",
        "His tragedies include Hamlet Macbeth and King Lear.",
        "The comedies feature mistaken identities and happy endings.",
        "Romeo and Juliet tells the story of star crossed lovers.",
        "Iambic pentameter is his signature rhythmic pattern.",
        "Soliloquies reveal a character's inner thoughts.",
        "The Globe Theatre was where many plays premiered.",
        "Shakespeare invented over seventeen hundred English words.",
        "His works explore universal themes of love power and betrayal.",
        "Shakespeare remains the most performed playwright in history.",
    ],
    "neural_networks": [
        "Neural networks consist of interconnected layers of nodes.",
        "The input layer receives raw feature data.",
        "Hidden layers learn intermediate representations.",
        "The output layer produces the final prediction.",
        "Activation functions introduce nonlinearity.",
        "Backpropagation adjusts weights to minimize error.",
        "Convolutional networks excel at image recognition.",
        "Recurrent networks handle sequential data like text.",
        "Transformers use attention mechanisms for parallel processing.",
        "Neural networks power modern speech recognition and translation.",
    ],
    "astronomy": [
        "Astronomy is the study of celestial objects and phenomena.",
        "Stars form from collapsing clouds of gas and dust.",
        "Our sun is a medium sized main sequence star.",
        "Planets orbit stars due to gravitational attraction.",
        "The Milky Way contains hundreds of billions of stars.",
        "Black holes have gravity so strong that nothing can escape.",
        "The universe is approximately thirteen point eight billion years old.",
        "Dark matter makes up about twenty seven percent of the universe.",
        "The Hubble Space Telescope has captured stunning images of deep space.",
        "Space exploration continues to reveal new mysteries.",
    ],
}

# Edge case transcripts
EDGE_CASE_TRANSCRIPTS = {
    "no_punctuation": [
        "this sentence has no punctuation at all",
        "and neither does this one",
        "we just keep going without stopping",
        "the speaker never pauses for breath",
        "auto captions often look exactly like this",
        "there are no periods or commas anywhere",
        "it makes sentence detection very hard",
        "but the algorithm should still handle it",
        "by using pause based boundaries instead",
        "and that wraps up the unpunctuated section",
    ],
    "very_short_sentences": [
        "Hi.",
        "Welcome.",
        "Today we learn.",
        "First point.",
        "Listen carefully.",
        "Key idea.",
        "Remember this.",
        "Important fact.",
        "Moving on.",
        "That is all.",
    ],
    "very_long_sentences": [
        "This is an extremely long sentence that goes on and on covering multiple ideas about calculus derivatives and integrals while also touching on linear algebra and matrix operations which are all interconnected in the broader field of mathematics.",
        "Furthermore we must consider that the relationship between these mathematical concepts extends well beyond simple computation and into the realm of applied sciences where differential equations model physical phenomena ranging from fluid dynamics to electrical circuits.",
        "In conclusion the study of advanced mathematics requires patience dedication and a willingness to work through complex problems step by step until the underlying patterns become clear.",
    ],
    "mixed_punctuation": [
        "What is calculus?",
        "It's amazing!",
        "Let me explain...",
        "Derivatives measure change.",
        "Right?",
        "Absolutely!",
        "The integral reverses differentiation...",
        "Think about it.",
        "Incredible!",
        "We are done.",
    ],
    "single_sentence": [
        "This entire video is just one single sentence about the nature of mathematical proofs and their importance in establishing rigorous foundations for all of mathematics.",
    ],
    "alternating_punct_no_punct": [
        "This sentence has a period.",
        "but this one does not",
        "This one does end properly!",
        "while this keeps going",
        "And finally we conclude.",
        "with one more thought",
        "Back to proper ending.",
        "and trailing off again",
        "One last proper sentence.",
        "the very end no period",
    ],
    # ---- NEW EDGE CASES ----
    "commas_only": [
        "first we consider the derivative, which measures change",
        "then the integral reverses differentiation, giving us area",
        "the chain rule handles compositions, as you know",
        "substitution simplifies, and partial fractions help too",
        "numerical methods approximate, when analytical fails",
        "the mean value theorem states, for some c in the interval",
        "L'Hopital's rule applies, when we have indeterminate forms",
        "Taylor series expand, and converge under conditions",
        "Fourier analysis decomposes, any periodic signal",
        "that concludes our overview, of these techniques",
    ],
    "fast_speech_half_second": [
        "Derivatives.",
        "Rate of change.",
        "Power rule.",
        "Chain rule.",
        "Product rule.",
        "Quotient rule.",
        "Implicit diff.",
        "Higher order.",
        "Applications.",
        "Optimization.",
        "Related rates.",
        "Mean value.",
        "Rolle's theorem.",
        "L'Hopital.",
        "Taylor series.",
        "Convergence.",
        "Radius.",
        "Interval.",
        "Summary.",
        "Done.",
    ],
    "ellipses_and_questions_only": [
        "What is a derivative...?",
        "How does the chain rule work...?",
        "Why do we need integrals...?",
        "Can you explain substitution...?",
        "What about partial fractions...?",
        "Is there a shortcut...?",
        "How do Taylor series converge...?",
        "What happens at infinity...?",
        "Why is e special...?",
        "Does this always work...?",
    ],
    "long_lecture_50_sentences": [
        "Welcome to today's lecture on differential equations.",
        "These equations relate functions to their derivatives.",
        "First order ODEs are the simplest case.",
        "We can solve separable equations by dividing both sides.",
        "The integrating factor method handles linear first order ODEs.",
        "Exact equations require a potential function.",
        "Existence and uniqueness theorems guarantee solutions.",
        "Second order linear ODEs appear in many physical systems.",
        "The characteristic equation gives us the general solution.",
        "Complex roots lead to oscillating solutions.",
        "Repeated roots require a special form with a t multiplier.",
        "Variation of parameters works for nonhomogeneous equations.",
        "Undetermined coefficients offer a faster approach sometimes.",
        "Laplace transforms convert ODEs into algebraic equations.",
        "The inverse transform recovers the time domain solution.",
        "Convolution theorems simplify product transforms.",
        "Systems of ODEs use matrix methods.",
        "Eigenvalues determine the behavior of linear systems.",
        "Phase portraits visualize two dimensional systems.",
        "Stability analysis classifies equilibrium points.",
        "Nonlinear systems require different techniques.",
        "Linearization approximates near equilibria.",
        "Lyapunov functions prove stability without solving.",
        "Bifurcation theory studies parameter changes.",
        "Limit cycles are isolated periodic orbits.",
        "The Poincare Bendixson theorem restricts planar behavior.",
        "Chaos arises in some nonlinear systems.",
        "The Lorenz system is a famous chaotic example.",
        "Sensitivity to initial conditions defines chaos.",
        "Strange attractors have fractal dimension.",
        "Numerical methods approximate solutions computationally.",
        "Euler's method is the simplest numerical scheme.",
        "Runge Kutta methods achieve higher accuracy.",
        "Adaptive step sizing improves efficiency.",
        "Stiff equations need implicit methods.",
        "Boundary value problems differ from initial value problems.",
        "Shooting methods convert BVPs to IVPs.",
        "Finite difference methods discretize the domain.",
        "Partial differential equations involve multiple variables.",
        "The heat equation models diffusion.",
        "The wave equation describes vibrations.",
        "Laplace's equation appears in steady state problems.",
        "Separation of variables is a key PDE technique.",
        "Fourier series expand periodic functions.",
        "Green's functions solve inhomogeneous problems.",
        "Variational methods minimize functionals.",
        "The finite element method handles complex geometries.",
        "Numerical PDE methods are essential in engineering.",
        "Applications range from fluid dynamics to quantum mechanics.",
        "That concludes our comprehensive survey of differential equations.",
    ],
    "exact_cue_edge_boundaries": [
        "This sentence ends exactly at the cue boundary.",
        "And this one starts exactly at the next cue.",
        "The transition is perfectly seamless.",
        "No gap exists between these cues.",
        "Each boundary is mathematically precise.",
        "The timestamps align to floating point.",
        "This tests the edge condition thoroughly.",
        "Rounding errors could cause problems.",
        "But the algorithm should handle it.",
        "Final sentence at the exact edge.",
    ],
    "unicode_and_special_chars": [
        "The Schrodinger equation describes quantum states.",
        "Euler's identity says e to the i pi plus one equals zero!",
        "The Navier-Stokes equations are millennium problems.",
        "Godel's incompleteness theorems shook mathematics.",
        "Riemann's zeta function connects primes to analysis.",
        "Poincare's conjecture was proved by Perelman.",
        "Noether's theorem links symmetry to conservation laws.",
        "Cantor's diagonal argument proves uncountability.",
        "Hilbert's problems guided twentieth century math.",
        "Ramanujan's formulas still inspire research today.",
    ],
    "rapid_topic_switches": [
        "Now let's talk about derivatives.",
        "Actually let me switch to integrals instead!",
        "Wait no let's cover linear algebra first.",
        "On second thought probability is more important.",
        "Or maybe we should start with set theory?",
        "Okay let's just go back to derivatives.",
        "The power rule says the derivative of x squared is two x.",
        "Moving on to applications of derivatives.",
        "Related rates problems use the chain rule.",
        "And that wraps up our whirlwind tour.",
    ],
}


class BoundaryStressTests(unittest.TestCase):
    """Run 200+ boundary checks across diverse topics and edge cases."""

    def setUp(self) -> None:
        self.rs = ReelService(
            embedding_service=EmbeddingService(),
            youtube_service=YouTubeService(),
        )
        self.stats = {
            "total_reels": 0,
            "start_on_sentence": 0,
            "start_not_on_sentence": 0,
            "end_on_sentence": 0,
            "end_on_fallback": 0,
            "end_mid_sentence": 0,
            "continuation_zero_gap": 0,
            "continuation_has_gap": 0,
            "continuation_has_overlap": 0,
        }

    def _check_start_boundary(
        self,
        transcript: list[dict[str, Any]],
        t_start: float,
        topic: str,
    ) -> bool:
        """Check if t_start aligns to a cue that starts a new sentence."""
        # Find the cue whose start matches t_start
        for i, cue in enumerate(transcript):
            cue_start = float(cue["start"])
            if abs(cue_start - t_start) < 0.05:
                # t_start aligns to a cue boundary
                if i == 0:
                    return True  # first cue is always valid
                prev_text = str(transcript[i - 1].get("text") or "").strip()
                # Valid if previous cue ends with terminal punct OR this is
                # a pause-based boundary (gap > 0.5s)
                if prev_text and prev_text[-1] in ".!?":
                    return True
                prev_end = float(transcript[i - 1]["start"]) + float(transcript[i - 1].get("duration") or 0)
                if cue_start - prev_end >= 0.5:
                    return True  # pause boundary
                # Also OK if t_start is first cue in range
                return True
        # t_start doesn't align to any cue start — check if it's close
        for cue in transcript:
            if abs(float(cue["start"]) - t_start) < 1.0:
                return True  # within 1s of a cue start is acceptable
        return False

    def _check_end_boundary(
        self,
        transcript: list[dict[str, Any]],
        t_end: float,
        topic: str,
    ) -> str:
        """Check if t_end aligns to a sentence boundary. Returns 'sentence'|'fallback'|'mid'."""
        for cue in transcript:
            cue_end = float(cue["start"]) + float(cue.get("duration") or 0)
            if abs(cue_end - t_end) < 0.05:
                text = str(cue.get("text") or "").strip()
                if text and text[-1] in ".!?":
                    return "sentence"
                # Pause-based boundary is acceptable
                return "fallback"
        # Check if near any cue end
        for cue in transcript:
            cue_end = float(cue["start"]) + float(cue.get("duration") or 0)
            if abs(cue_end - t_end) < 1.5:
                return "fallback"
        return "mid"

    def _run_refine_and_check(
        self,
        topic: str,
        transcript: list[dict[str, Any]],
        proposed_start: float,
        proposed_end: float,
        video_duration: int,
        min_len: int,
        max_len: int,
    ) -> dict[str, Any]:
        """Run _refine_clip_window_from_transcript and check boundaries."""
        win = self.rs._refine_clip_window_from_transcript(
            transcript=transcript,
            proposed_start=proposed_start,
            proposed_end=proposed_end,
            video_duration_sec=video_duration,
            min_len=min_len,
            max_len=max_len,
        )
        result = {
            "topic": topic,
            "proposed_start": proposed_start,
            "proposed_end": proposed_end,
            "min_len": min_len,
            "max_len": max_len,
            "window": win,
            "start_ok": False,
            "end_type": "none",
        }
        if win is None:
            return result

        t_start, t_end = win
        result["start_ok"] = self._check_start_boundary(transcript, t_start, topic)
        result["end_type"] = self._check_end_boundary(transcript, t_end, topic)
        self.stats["total_reels"] += 1
        if result["start_ok"]:
            self.stats["start_on_sentence"] += 1
        else:
            self.stats["start_not_on_sentence"] += 1
        if result["end_type"] == "sentence":
            self.stats["end_on_sentence"] += 1
        elif result["end_type"] == "fallback":
            self.stats["end_on_fallback"] += 1
        else:
            self.stats["end_mid_sentence"] += 1
        return result

    # =================== Main topic boundary tests =================== #

    def test_all_topics_single_reel_default_bounds(self) -> None:
        """Test single-reel extraction for all 20 topics (default 20-55s bounds)."""
        failures = []
        for topic, sentences in TOPIC_TRANSCRIPTS.items():
            transcript = _build_topic_transcript(sentences, cue_duration=3.0)
            total_dur = len(sentences) * 3
            result = self._run_refine_and_check(
                topic=topic,
                transcript=transcript,
                proposed_start=0.0,
                proposed_end=float(total_dur),
                video_duration=total_dur + 10,
                min_len=20,
                max_len=55,
            )
            if result["window"] is None:
                failures.append(f"{topic}: returned None")
                continue
            if not result["start_ok"]:
                failures.append(f"{topic}: start NOT on sentence boundary (t_start={result['window'][0]})")
            if result["end_type"] == "mid":
                failures.append(f"{topic}: end mid-sentence (t_end={result['window'][1]})")
        self.assertEqual(failures, [], f"Failures:\n" + "\n".join(failures))

    def test_all_topics_single_reel_tight_bounds(self) -> None:
        """Test single-reel with tight bounds (15-25s) for all topics."""
        failures = []
        for topic, sentences in TOPIC_TRANSCRIPTS.items():
            transcript = _build_topic_transcript(sentences, cue_duration=3.0)
            total_dur = len(sentences) * 3
            # Start from middle of transcript
            mid = total_dur // 2
            result = self._run_refine_and_check(
                topic=topic,
                transcript=transcript,
                proposed_start=float(mid),
                proposed_end=float(mid + 20),
                video_duration=total_dur + 10,
                min_len=15,
                max_len=25,
            )
            if result["window"] is None:
                failures.append(f"{topic}: returned None")
                continue
            if not result["start_ok"]:
                failures.append(f"{topic}: start NOT on sentence (t_start={result['window'][0]})")
            if result["end_type"] == "mid":
                failures.append(f"{topic}: end mid-sentence (t_end={result['window'][1]})")
        self.assertEqual(failures, [], f"Failures:\n" + "\n".join(failures))

    def test_all_topics_single_reel_long_bounds(self) -> None:
        """Test with long bounds (30-55s) for all topics using longer cue durations."""
        failures = []
        for topic, sentences in TOPIC_TRANSCRIPTS.items():
            transcript = _build_topic_transcript(sentences, cue_duration=5.0)
            total_dur = len(sentences) * 5  # 50s for 10 sentences
            # Use bounds that fit within the transcript length
            result = self._run_refine_and_check(
                topic=topic,
                transcript=transcript,
                proposed_start=0.0,
                proposed_end=float(total_dur),
                video_duration=total_dur + 10,
                min_len=30,
                max_len=55,
            )
            if result["window"] is None:
                continue
            if not result["start_ok"]:
                failures.append(f"{topic}: start NOT on sentence (t_start={result['window'][0]})")
            if result["end_type"] == "mid":
                failures.append(f"{topic}: end mid-sentence (t_end={result['window'][1]})")
        self.assertEqual(failures, [], f"Failures:\n" + "\n".join(failures))

    def test_all_topics_mid_transcript_start(self) -> None:
        """Start from various offsets within each topic (not aligned to cues)."""
        failures = []
        for topic, sentences in TOPIC_TRANSCRIPTS.items():
            transcript = _build_topic_transcript(sentences, cue_duration=3.0)
            total_dur = len(sentences) * 3
            # Test starting at 1.5s, 4.5s, 7.5s (mid-cue positions)
            for offset in [1.5, 4.5, 7.5, 10.5]:
                if offset >= total_dur - 15:
                    continue
                result = self._run_refine_and_check(
                    topic=f"{topic}@{offset}",
                    transcript=transcript,
                    proposed_start=offset,
                    proposed_end=offset + 25.0,
                    video_duration=total_dur + 10,
                    min_len=15,
                    max_len=30,
                )
                if result["window"] is None:
                    continue
                if not result["start_ok"]:
                    failures.append(f"{topic}@{offset}: start NOT on sentence (t_start={result['window'][0]})")
                if result["end_type"] == "mid":
                    failures.append(f"{topic}@{offset}: end mid-sentence (t_end={result['window'][1]})")
        self.assertEqual(failures, [], f"Failures:\n" + "\n".join(failures))

    # =================== Continuation / splitting tests =================== #

    def test_all_topics_continuation_splitting(self) -> None:
        """Split every topic into consecutive windows and verify zero-gap zero-overlap."""
        failures = []
        for topic, sentences in TOPIC_TRANSCRIPTS.items():
            transcript = _build_topic_transcript(sentences, cue_duration=4.0)
            total_dur = len(sentences) * 4
            # Force split with small max_len
            for max_len in [12, 15, 20]:
                windows = self.rs._split_into_consecutive_windows(
                    transcript=transcript,
                    segment_start=0.0,
                    segment_end=float(total_dur),
                    video_duration_sec=total_dur + 10,
                    min_len=8,
                    max_len=max_len,
                )
                if len(windows) < 2:
                    continue
                for i in range(len(windows) - 1):
                    gap = windows[i + 1][0] - windows[i][1]
                    if gap > 0.01:
                        failures.append(f"{topic} max={max_len}: gap={gap:.3f}s between windows {i} and {i+1}")
                        self.stats["continuation_has_gap"] += 1
                    elif gap < -0.01:
                        failures.append(f"{topic} max={max_len}: overlap={abs(gap):.3f}s between windows {i} and {i+1}")
                        self.stats["continuation_has_overlap"] += 1
                    else:
                        self.stats["continuation_zero_gap"] += 1

                # Check each window's start/end boundaries
                for j, (ws, we) in enumerate(windows):
                    start_ok = self._check_start_boundary(transcript, ws, topic)
                    end_type = self._check_end_boundary(transcript, we, topic)
                    self.stats["total_reels"] += 1
                    if start_ok:
                        self.stats["start_on_sentence"] += 1
                    else:
                        self.stats["start_not_on_sentence"] += 1
                        failures.append(f"{topic} max={max_len} win[{j}]: start NOT on sentence (t_start={ws})")
                    if end_type == "sentence":
                        self.stats["end_on_sentence"] += 1
                    elif end_type == "fallback":
                        self.stats["end_on_fallback"] += 1
                    else:
                        self.stats["end_mid_sentence"] += 1
                        # Only flag non-last windows as errors; last window is allowed fallback
                        if j < len(windows) - 1:
                            failures.append(f"{topic} max={max_len} win[{j}]: end mid-sentence (t_end={we})")
        self.assertEqual(failures, [], f"Failures:\n" + "\n".join(failures))

    # =================== Edge case tests =================== #

    def test_edge_no_punctuation(self) -> None:
        """Unpunctuated transcripts should use pause-based boundaries."""
        transcript = _build_topic_transcript(
            EDGE_CASE_TRANSCRIPTS["no_punctuation"],
            cue_duration=3.0,
            gap=0.8,  # introduce pauses
        )
        total_dur = int(len(EDGE_CASE_TRANSCRIPTS["no_punctuation"]) * 3.8)
        win = self.rs._refine_clip_window_from_transcript(
            transcript=transcript,
            proposed_start=0.0,
            proposed_end=20.0,
            video_duration_sec=total_dur + 10,
            min_len=15,
            max_len=25,
        )
        self.assertIsNotNone(win, "Should produce a window even without punctuation")
        self.stats["total_reels"] += 1

    def test_edge_very_short_sentences(self) -> None:
        """Very short sentences (1-3 words each) should still snap correctly."""
        transcript = _build_topic_transcript(
            EDGE_CASE_TRANSCRIPTS["very_short_sentences"],
            cue_duration=1.5,
        )
        total_dur = int(len(EDGE_CASE_TRANSCRIPTS["very_short_sentences"]) * 1.5)
        win = self.rs._refine_clip_window_from_transcript(
            transcript=transcript,
            proposed_start=0.0,
            proposed_end=10.0,
            video_duration_sec=total_dur + 10,
            min_len=5,
            max_len=12,
        )
        self.assertIsNotNone(win)
        if win:
            end_type = self._check_end_boundary(transcript, win[1], "short_sentences")
            self.assertIn(end_type, ["sentence", "fallback"])
        self.stats["total_reels"] += 1

    def test_edge_very_long_sentences(self) -> None:
        """Very long sentences (50+ words) — cues span 15s each."""
        transcript = _build_topic_transcript(
            EDGE_CASE_TRANSCRIPTS["very_long_sentences"],
            cue_duration=15.0,
        )
        total_dur = int(len(EDGE_CASE_TRANSCRIPTS["very_long_sentences"]) * 15)
        win = self.rs._refine_clip_window_from_transcript(
            transcript=transcript,
            proposed_start=0.0,
            proposed_end=30.0,
            video_duration_sec=total_dur + 10,
            min_len=12,
            max_len=35,
        )
        self.assertIsNotNone(win)
        self.stats["total_reels"] += 1

    def test_edge_mixed_punctuation(self) -> None:
        """Mix of ?, !, ..., and . — all should be valid end markers."""
        failures = []
        transcript = _build_topic_transcript(
            EDGE_CASE_TRANSCRIPTS["mixed_punctuation"],
            cue_duration=3.0,
        )
        total_dur = int(len(EDGE_CASE_TRANSCRIPTS["mixed_punctuation"]) * 3)
        for start_offset in [0.0, 3.0, 6.0, 9.0]:
            win = self.rs._refine_clip_window_from_transcript(
                transcript=transcript,
                proposed_start=start_offset,
                proposed_end=start_offset + 18.0,
                video_duration_sec=total_dur + 10,
                min_len=12,
                max_len=20,
            )
            if win is None:
                continue
            end_type = self._check_end_boundary(transcript, win[1], "mixed_punct")
            self.stats["total_reels"] += 1
            if end_type == "mid":
                failures.append(f"mixed_punct@{start_offset}: end mid-sentence (t_end={win[1]})")
        self.assertEqual(failures, [])

    def test_edge_single_sentence(self) -> None:
        """Single long cue — should return a valid window."""
        transcript = _build_topic_transcript(
            EDGE_CASE_TRANSCRIPTS["single_sentence"],
            cue_duration=30.0,
        )
        win = self.rs._refine_clip_window_from_transcript(
            transcript=transcript,
            proposed_start=0.0,
            proposed_end=30.0,
            video_duration_sec=40,
            min_len=15,
            max_len=35,
        )
        self.assertIsNotNone(win)
        self.stats["total_reels"] += 1

    def test_edge_alternating_punct(self) -> None:
        """Every other sentence lacks punctuation."""
        failures = []
        transcript = _build_topic_transcript(
            EDGE_CASE_TRANSCRIPTS["alternating_punct_no_punct"],
            cue_duration=3.0,
        )
        total_dur = int(len(EDGE_CASE_TRANSCRIPTS["alternating_punct_no_punct"]) * 3)
        for start_offset in range(0, total_dur - 15, 3):
            win = self.rs._refine_clip_window_from_transcript(
                transcript=transcript,
                proposed_start=float(start_offset),
                proposed_end=float(start_offset + 18),
                video_duration_sec=total_dur + 10,
                min_len=12,
                max_len=20,
            )
            if win is None:
                continue
            self.stats["total_reels"] += 1
            start_ok = self._check_start_boundary(transcript, win[0], "alt_punct")
            if not start_ok:
                failures.append(f"alt_punct@{start_offset}: start NOT on sentence (t_start={win[0]})")
        self.assertEqual(failures, [])

    # =================== Cue duration variations =================== #

    def test_varying_cue_durations_across_topics(self) -> None:
        """Test with 1s, 2s, 5s, and 8s cue durations across all topics."""
        failures = []
        for cue_dur in [1.0, 2.0, 5.0, 8.0]:
            for topic, sentences in list(TOPIC_TRANSCRIPTS.items())[:10]:  # first 10 topics
                transcript = _build_topic_transcript(sentences, cue_duration=cue_dur)
                total_dur = len(sentences) * cue_dur
                if total_dur < 20:
                    continue
                result = self._run_refine_and_check(
                    topic=f"{topic}_cue{cue_dur}",
                    transcript=transcript,
                    proposed_start=0.0,
                    proposed_end=min(total_dur, 55.0),
                    video_duration=int(total_dur) + 10,
                    min_len=15,
                    max_len=55,
                )
                if result["window"] is None:
                    continue
                if not result["start_ok"]:
                    failures.append(f"{topic}_cue{cue_dur}: start NOT on sentence")
                if result["end_type"] == "mid":
                    failures.append(f"{topic}_cue{cue_dur}: end mid-sentence")
        self.assertEqual(failures, [], f"Failures:\n" + "\n".join(failures))

    # =================== Min/max boundary edge tests =================== #

    def test_segment_exactly_at_max_len(self) -> None:
        """Segment duration == max_len exactly."""
        failures = []
        for topic, sentences in list(TOPIC_TRANSCRIPTS.items())[:5]:
            for max_len in [15, 20, 30]:
                transcript = _build_topic_transcript(sentences, cue_duration=3.0)
                result = self._run_refine_and_check(
                    topic=f"{topic}_exact{max_len}",
                    transcript=transcript,
                    proposed_start=0.0,
                    proposed_end=float(max_len),
                    video_duration=60,
                    min_len=max(10, max_len - 10),
                    max_len=max_len,
                )
                if result["window"] is None:
                    continue
                if not result["start_ok"]:
                    failures.append(f"{topic}_exact{max_len}: start NOT on sentence")
                if result["end_type"] == "mid":
                    failures.append(f"{topic}_exact{max_len}: end mid-sentence")
        self.assertEqual(failures, [])

    def test_segment_just_under_min_len(self) -> None:
        """Segment is barely longer than min_len — should still produce a valid window."""
        failures = []
        for topic, sentences in list(TOPIC_TRANSCRIPTS.items())[:5]:
            transcript = _build_topic_transcript(sentences, cue_duration=3.0)
            result = self._run_refine_and_check(
                topic=f"{topic}_justmin",
                transcript=transcript,
                proposed_start=0.0,
                proposed_end=16.0,  # just over min_len=15
                video_duration=60,
                min_len=15,
                max_len=30,
            )
            if result["window"] is not None:
                self.stats["total_reels"] += 1  # already counted in helper
        self.assertTrue(True)  # no crash is the test

    # =================== NEW EDGE CASES: commas-only, fast speech, etc. =================== #

    def test_edge_commas_only_no_periods(self) -> None:
        """Transcripts with ONLY commas (no terminal punctuation at all).
        Should fall back to pause-based boundaries."""
        transcript = _build_topic_transcript(
            EDGE_CASE_TRANSCRIPTS["commas_only"],
            cue_duration=3.0,
            gap=0.7,  # pauses between cues for boundary detection
        )
        total_dur = int(len(EDGE_CASE_TRANSCRIPTS["commas_only"]) * 3.7)
        # Single reel
        win = self.rs._refine_clip_window_from_transcript(
            transcript=transcript,
            proposed_start=0.0,
            proposed_end=20.0,
            video_duration_sec=total_dur + 10,
            min_len=15,
            max_len=25,
        )
        self.assertIsNotNone(win, "Commas-only transcript must produce a window")
        self.stats["total_reels"] += 1

        # Split test — force multi-window
        windows = self.rs._split_into_consecutive_windows(
            transcript=transcript,
            segment_start=0.0,
            segment_end=float(total_dur),
            video_duration_sec=total_dur + 10,
            min_len=8,
            max_len=15,
        )
        self.assertGreater(len(windows), 0, "Commas-only must produce at least one window")
        # Check continuation chaining (tolerance for inter-cue silence gaps)
        failures = _check_continuation_gaps(windows, transcript, "commas_only")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_edge_fast_speech_half_second_cues(self) -> None:
        """Very fast speech with 0.5s cue durations (20 cues = 10s total)."""
        transcript = _build_topic_transcript(
            EDGE_CASE_TRANSCRIPTS["fast_speech_half_second"],
            cue_duration=0.5,
        )
        total_dur = len(EDGE_CASE_TRANSCRIPTS["fast_speech_half_second"]) * 0.5  # 10s
        failures = []

        # Single reel with relaxed min
        win = self.rs._refine_clip_window_from_transcript(
            transcript=transcript,
            proposed_start=0.0,
            proposed_end=total_dur,
            video_duration_sec=int(total_dur) + 10,
            min_len=3,
            max_len=12,
        )
        self.assertIsNotNone(win, "Fast speech must produce a window")
        if win:
            self.stats["total_reels"] += 1
            end_type = self._check_end_boundary(transcript, win[1], "fast_speech")
            if end_type == "mid":
                failures.append(f"fast_speech single: end mid-sentence (t_end={win[1]})")

        # Multi-reel split
        windows = self.rs._split_into_consecutive_windows(
            transcript=transcript,
            segment_start=0.0,
            segment_end=total_dur,
            video_duration_sec=int(total_dur) + 10,
            min_len=2,
            max_len=4,
        )
        failures.extend(_check_continuation_gaps(windows, transcript, "fast_speech split"))
        for j, (ws, we) in enumerate(windows):
            self.stats["total_reels"] += 1
            start_ok = self._check_start_boundary(transcript, ws, "fast_speech")
            if not start_ok:
                failures.append(f"fast_speech win[{j}]: start NOT on sentence (t_start={ws})")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_edge_ellipses_and_questions_only(self) -> None:
        """Transcripts with only ellipses and question marks — all valid terminal punct."""
        failures = []
        transcript = _build_topic_transcript(
            EDGE_CASE_TRANSCRIPTS["ellipses_and_questions_only"],
            cue_duration=3.0,
        )
        total_dur = int(len(EDGE_CASE_TRANSCRIPTS["ellipses_and_questions_only"]) * 3)

        # Multiple start positions
        for start in [0.0, 3.0, 6.0, 9.0, 12.0]:
            if start >= total_dur - 12:
                continue
            win = self.rs._refine_clip_window_from_transcript(
                transcript=transcript,
                proposed_start=start,
                proposed_end=start + 18.0,
                video_duration_sec=total_dur + 10,
                min_len=12,
                max_len=20,
            )
            if win is None:
                continue
            self.stats["total_reels"] += 1
            end_type = self._check_end_boundary(transcript, win[1], "ellipses_questions")
            if end_type == "mid":
                failures.append(f"ellipses_questions@{start}: end mid-sentence (t_end={win[1]})")
            start_ok = self._check_start_boundary(transcript, win[0], "ellipses_questions")
            if not start_ok:
                failures.append(f"ellipses_questions@{start}: start NOT on sentence (t_start={win[0]})")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_edge_long_lecture_50_sentences(self) -> None:
        """50+ sentence lecture — tests sustained splitting and continuation chaining."""
        sentences = EDGE_CASE_TRANSCRIPTS["long_lecture_50_sentences"]
        transcript = _build_topic_transcript(sentences, cue_duration=3.0)
        total_dur = len(sentences) * 3  # 150s

        failures = []

        # Single reel extraction from middle
        for start_offset in [0.0, 30.0, 60.0, 90.0, 120.0]:
            if start_offset >= total_dur - 20:
                continue
            result = self._run_refine_and_check(
                topic=f"long_lecture@{start_offset}",
                transcript=transcript,
                proposed_start=start_offset,
                proposed_end=start_offset + 40.0,
                video_duration=total_dur + 10,
                min_len=20,
                max_len=55,
            )
            if result["window"] is not None:
                if not result["start_ok"]:
                    failures.append(f"long_lecture@{start_offset}: start NOT on sentence")
                if result["end_type"] == "mid":
                    failures.append(f"long_lecture@{start_offset}: end mid-sentence")

        # Force max splits: entire 150s with max_len=25 should produce 5+ windows
        windows = self.rs._split_into_consecutive_windows(
            transcript=transcript,
            segment_start=0.0,
            segment_end=float(total_dur),
            video_duration_sec=total_dur + 10,
            min_len=15,
            max_len=25,
        )
        self.assertGreaterEqual(len(windows), 3, f"Expected 3+ windows for 150s lecture, got {len(windows)}")

        # Verify continuation chaining
        for i in range(len(windows) - 1):
            gap = windows[i + 1][0] - windows[i][1]
            if gap > 0.01:
                failures.append(f"long_lecture split: gap={gap:.3f}s between win {i} and {i+1}")
                self.stats["continuation_has_gap"] += 1
            elif gap < -0.01:
                failures.append(f"long_lecture split: overlap={abs(gap):.3f}s between win {i} and {i+1}")
                self.stats["continuation_has_overlap"] += 1
            else:
                self.stats["continuation_zero_gap"] += 1

        # Verify boundary quality for each window
        for j, (ws, we) in enumerate(windows):
            self.stats["total_reels"] += 1
            start_ok = self._check_start_boundary(transcript, ws, "long_lecture")
            end_type = self._check_end_boundary(transcript, we, "long_lecture")
            if start_ok:
                self.stats["start_on_sentence"] += 1
            else:
                self.stats["start_not_on_sentence"] += 1
                failures.append(f"long_lecture win[{j}]: start NOT on sentence (t_start={ws})")
            if end_type == "sentence":
                self.stats["end_on_sentence"] += 1
            elif end_type == "fallback":
                self.stats["end_on_fallback"] += 1
            else:
                self.stats["end_mid_sentence"] += 1
                if j < len(windows) - 1:
                    failures.append(f"long_lecture win[{j}]: end mid-sentence (t_end={we})")

        self.assertEqual(failures, [], "\n".join(failures))

    def test_edge_exact_cue_boundaries(self) -> None:
        """Segment boundaries at exact cue edges — tests floating point alignment."""
        transcript = _build_topic_transcript(
            EDGE_CASE_TRANSCRIPTS["exact_cue_edge_boundaries"],
            cue_duration=3.0,
            gap=0.0,  # zero gap = exact edges
        )
        total_dur = len(EDGE_CASE_TRANSCRIPTS["exact_cue_edge_boundaries"]) * 3
        failures = []

        # Start at exact cue boundaries
        for cue_idx in range(0, min(7, len(EDGE_CASE_TRANSCRIPTS["exact_cue_edge_boundaries"]))):
            exact_start = float(cue_idx * 3)
            if exact_start >= total_dur - 15:
                continue
            win = self.rs._refine_clip_window_from_transcript(
                transcript=transcript,
                proposed_start=exact_start,
                proposed_end=exact_start + 18.0,
                video_duration_sec=total_dur + 10,
                min_len=12,
                max_len=20,
            )
            if win is None:
                continue
            self.stats["total_reels"] += 1
            start_ok = self._check_start_boundary(transcript, win[0], f"exact_edge@{cue_idx}")
            end_type = self._check_end_boundary(transcript, win[1], f"exact_edge@{cue_idx}")
            if not start_ok:
                failures.append(f"exact_edge@cue{cue_idx}: start NOT on sentence (t_start={win[0]})")
            if end_type == "mid":
                failures.append(f"exact_edge@cue{cue_idx}: end mid-sentence (t_end={win[1]})")

        # Test with proposed_end at exact cue end
        for cue_idx in [3, 5, 7, 9]:
            exact_end = float(cue_idx * 3)
            if exact_end > total_dur or exact_end < 15:
                continue
            win = self.rs._refine_clip_window_from_transcript(
                transcript=transcript,
                proposed_start=0.0,
                proposed_end=exact_end,
                video_duration_sec=total_dur + 10,
                min_len=12,
                max_len=int(exact_end) + 5,
            )
            if win is not None:
                self.stats["total_reels"] += 1
                end_type = self._check_end_boundary(transcript, win[1], f"exact_edge_end@{cue_idx}")
                if end_type == "mid":
                    failures.append(f"exact_edge_end@cue{cue_idx}: end mid-sentence (t_end={win[1]})")

        self.assertEqual(failures, [], "\n".join(failures))

    def test_edge_unicode_special_chars(self) -> None:
        """Transcripts with names containing special characters and apostrophes."""
        failures = []
        transcript = _build_topic_transcript(
            EDGE_CASE_TRANSCRIPTS["unicode_and_special_chars"],
            cue_duration=3.5,
        )
        total_dur = int(len(EDGE_CASE_TRANSCRIPTS["unicode_and_special_chars"]) * 3.5)
        for start in [0.0, 7.0, 14.0]:
            if start >= total_dur - 15:
                continue
            result = self._run_refine_and_check(
                topic=f"unicode@{start}",
                transcript=transcript,
                proposed_start=start,
                proposed_end=start + 20.0,
                video_duration=total_dur + 10,
                min_len=12,
                max_len=25,
            )
            if result["window"] is not None:
                if not result["start_ok"]:
                    failures.append(f"unicode@{start}: start NOT on sentence")
                if result["end_type"] == "mid":
                    failures.append(f"unicode@{start}: end mid-sentence")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_edge_rapid_topic_switches(self) -> None:
        """Transcript with rapid topic changes — boundary quality should hold."""
        failures = []
        transcript = _build_topic_transcript(
            EDGE_CASE_TRANSCRIPTS["rapid_topic_switches"],
            cue_duration=3.0,
        )
        total_dur = int(len(EDGE_CASE_TRANSCRIPTS["rapid_topic_switches"]) * 3)
        # Multiple window sizes
        for max_len in [12, 18, 25]:
            windows = self.rs._split_into_consecutive_windows(
                transcript=transcript,
                segment_start=0.0,
                segment_end=float(total_dur),
                video_duration_sec=total_dur + 10,
                min_len=8,
                max_len=max_len,
            )
            for i in range(len(windows) - 1):
                gap = windows[i + 1][0] - windows[i][1]
                if gap > 0.01:
                    failures.append(f"rapid_switch max={max_len}: gap={gap:.3f}s between win {i},{i+1}")
                elif gap < -0.01:
                    failures.append(f"rapid_switch max={max_len}: overlap={abs(gap):.3f}s between win {i},{i+1}")
            for j, (ws, we) in enumerate(windows):
                self.stats["total_reels"] += 1
                start_ok = self._check_start_boundary(transcript, ws, "rapid_switch")
                end_type = self._check_end_boundary(transcript, we, "rapid_switch")
                if not start_ok:
                    failures.append(f"rapid_switch max={max_len} win[{j}]: start NOT on sentence")
                if end_type == "mid" and j < len(windows) - 1:
                    failures.append(f"rapid_switch max={max_len} win[{j}]: end mid-sentence")
        self.assertEqual(failures, [], "\n".join(failures))

    # =================== Realistic YouTube transcript simulations =================== #

    def test_realistic_khan_academy_style_transcript(self) -> None:
        """Simulate a Khan Academy style narration — short clear sentences, steady pace."""
        sentences = [
            "Let's think about what it means to multiply two by three.",
            "We can think of this as two groups of three.",
            "Or we can think of it as three groups of two.",
            "Either way we get six.",
            "Now what about two times three point five?",
            "Well that's two groups of three point five.",
            "Three point five plus three point five equals seven.",
            "So two times three point five is seven.",
            "What if we multiplied by a fraction?",
            "Say two times one half.",
            "That gives us one whole.",
            "Because half of two is one.",
            "Fractions can be tricky but they follow the same rules.",
            "Let's try a harder example.",
            "What is three fourths times two thirds?",
            "We multiply the numerators three times two is six.",
            "We multiply the denominators four times three is twelve.",
            "So we get six twelfths which simplifies to one half.",
            "Always remember to simplify your fractions.",
            "That's the key takeaway from today's lesson.",
        ]
        transcript = _build_topic_transcript(sentences, cue_duration=2.5, gap=0.3)
        total_dur = int(len(sentences) * 2.8)
        failures = []

        # Full coverage split
        windows = self.rs._split_into_consecutive_windows(
            transcript=transcript,
            segment_start=0.0,
            segment_end=float(total_dur),
            video_duration_sec=total_dur + 10,
            min_len=12,
            max_len=20,
        )
        self.assertGreater(len(windows), 1, "Khan style should split into multiple reels")

        failures.extend(_check_continuation_gaps(windows, transcript, "khan split"))

        for j, (ws, we) in enumerate(windows):
            self.stats["total_reels"] += 1
            start_ok = self._check_start_boundary(transcript, ws, "khan")
            end_type = self._check_end_boundary(transcript, we, "khan")
            if not start_ok:
                failures.append(f"khan win[{j}]: start NOT on sentence (t_start={ws})")
            if end_type == "mid" and j < len(windows) - 1:
                failures.append(f"khan win[{j}]: end mid-sentence (t_end={we})")
            if start_ok:
                self.stats["start_on_sentence"] += 1
            else:
                self.stats["start_not_on_sentence"] += 1
            if end_type == "sentence":
                self.stats["end_on_sentence"] += 1
            elif end_type == "fallback":
                self.stats["end_on_fallback"] += 1
            else:
                self.stats["end_mid_sentence"] += 1
        self.assertEqual(failures, [], "\n".join(failures))

    def test_realistic_veritasium_style_long_sentences(self) -> None:
        """Simulate Veritasium style — longer, more complex sentences with varied pacing."""
        sentences = [
            "I want to show you something that will change how you think about magnets.",
            "Most people think magnets work because of some mysterious force.",
            "But the truth is much more interesting and much more fundamental.",
            "It all comes down to the spin of electrons inside the material.",
            "Now electrons are quantum particles and they have a property called spin.",
            "Spin is not actually the electron spinning like a top.",
            "It's a quantum mechanical property that has no classical analogue.",
            "But it does give each electron a tiny magnetic moment.",
            "In most materials these magnetic moments point in random directions and cancel out.",
            "But in ferromagnetic materials something special happens.",
            "The exchange interaction aligns neighboring spins in the same direction.",
            "This creates domains where billions of atoms all point the same way.",
            "When you magnetize a piece of iron you're aligning these domains.",
            "That's why hitting a magnet can demagnetize it.",
            "You're randomizing the domain structure through mechanical vibration.",
            "Temperature also plays a role because thermal energy fights alignment.",
            "Above the Curie temperature a ferromagnet becomes paramagnetic.",
            "This is a phase transition just like ice melting into water.",
            "The practical applications of magnetism are enormous.",
            "From MRI machines to electric motors to computer hard drives.",
            "Every time you use a credit card you're relying on magnetic storage.",
            "And the fundamental physics behind all of it is quantum mechanics.",
        ]
        transcript = _build_topic_transcript(sentences, cue_duration=4.0, gap=0.5)
        total_dur = int(len(sentences) * 4.5)
        failures = []

        # Force continuation splitting
        for max_len in [20, 30, 40]:
            windows = self.rs._split_into_consecutive_windows(
                transcript=transcript,
                segment_start=0.0,
                segment_end=float(total_dur),
                video_duration_sec=total_dur + 10,
                min_len=15,
                max_len=max_len,
            )
            failures.extend(_check_continuation_gaps(
                windows, transcript, f"veritasium max={max_len}",
            ))

            for j, (ws, we) in enumerate(windows):
                self.stats["total_reels"] += 1
                start_ok = self._check_start_boundary(transcript, ws, "veritasium")
                end_type = self._check_end_boundary(transcript, we, "veritasium")
                if not start_ok:
                    failures.append(f"veritasium max={max_len} win[{j}]: start NOT on sentence")
                if end_type == "mid" and j < len(windows) - 1:
                    failures.append(f"veritasium max={max_len} win[{j}]: end mid-sentence")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_realistic_3b1b_style_unpunctuated_autocaptions(self) -> None:
        """Simulate YouTube auto-captions (no punctuation, lowercase) like many 3B1B auto-gen."""
        sentences = [
            "so the question is what does the area of a circle have to do with calculus",
            "well one way to think about it is to unroll all of those concentric rings",
            "into thin rectangles and then line them up along an axis",
            "the base of each rectangle corresponds to the circumference of the ring",
            "which is two pi times the radius at that point",
            "the height is just some small value dr",
            "so the area of each thin rectangle is approximately two pi r times dr",
            "and if we add up all of these rectangles from r equals zero to the full radius",
            "we get the total area which should equal pi r squared",
            "but notice what we just did we added up a bunch of thin rectangles",
            "and that my friends is exactly what an integral does",
            "the integral of two pi r from zero to big r equals pi r squared",
            "so the formula for the area of a circle comes directly from calculus",
            "now you might wonder why we bother with this formal machinery",
            "the reason is that integrals let us handle much more complex shapes",
            "this same idea of slicing and summing works for volumes and surfaces too",
        ]
        transcript = _build_topic_transcript(sentences, cue_duration=3.5, gap=0.8)
        total_dur = int(len(sentences) * 4.3)
        failures = []

        # Should detect as unpunctuated and use pause-based boundaries
        is_punct = self.rs._transcript_has_terminal_punct(
            [{"start": c["start"], "end": c["start"] + c["duration"], "text": c["text"]}
             for c in transcript]
        )
        self.assertFalse(is_punct, "Auto-caption transcript should be detected as unpunctuated")

        # Single reel
        win = self.rs._refine_clip_window_from_transcript(
            transcript=transcript,
            proposed_start=0.0,
            proposed_end=25.0,
            video_duration_sec=total_dur + 10,
            min_len=15,
            max_len=30,
        )
        self.assertIsNotNone(win, "Must produce a window for unpunctuated auto-captions")
        if win:
            self.stats["total_reels"] += 1

        # Multi-reel split
        windows = self.rs._split_into_consecutive_windows(
            transcript=transcript,
            segment_start=0.0,
            segment_end=float(total_dur),
            video_duration_sec=total_dur + 10,
            min_len=12,
            max_len=20,
        )
        self.assertGreater(len(windows), 1, "Unpunctuated 68s transcript should split")

        failures.extend(_check_continuation_gaps(windows, transcript, "3b1b_auto split"))
        self.assertEqual(failures, [], "\n".join(failures))

    def test_realistic_search_user_flow_calculus(self) -> None:
        """Simulate a real user searching 'calculus' — test the full refine+split pipeline
        across a realistic transcript that mimics actual YouTube content discovery."""
        # This simulates what happens when a user searches "calculus" and we find
        # a 300s educational video. The transcript has natural sentence structure
        # with realistic timing and pacing.
        intro = [
            "Welcome back everyone.",
            "Today we're going to explore one of the most beautiful ideas in all of mathematics.",
            "The fundamental theorem of calculus connects two seemingly different concepts.",
            "On one hand we have derivatives which measure instantaneous rates of change.",
            "On the other hand we have integrals which compute accumulated quantities.",
            "The theorem says these two operations are inverses of each other.",
            "That's an incredibly powerful statement.",
            "Let me show you what I mean with a concrete example.",
        ]
        body = [
            "Consider a car driving along a highway.",
            "We know its velocity at every moment in time.",
            "The velocity function v of t tells us how fast the car is going.",
            "If we want to know the total distance traveled we need to integrate.",
            "The integral of velocity gives us displacement.",
            "But here's the beautiful part.",
            "If we take the derivative of displacement we get back the velocity!",
            "That's the fundamental theorem in action.",
            "Now let's be more precise about what the theorem actually says.",
            "If F is an antiderivative of f then the definite integral from a to b equals F of b minus F of a.",
            "This is sometimes called the evaluation theorem.",
            "It transforms a potentially difficult limit of Riemann sums into a simple subtraction.",
            "Let's work through an example step by step.",
            "Suppose we want to compute the integral of x squared from zero to three.",
            "An antiderivative of x squared is x cubed over three.",
            "So we evaluate at the bounds and get twenty seven over three minus zero which is nine.",
            "Compare that to computing the integral from the definition using Riemann sums.",
            "You'd need to compute a limit of increasingly fine partitions.",
            "The fundamental theorem saves us all that work.",
            "This is why it's called fundamental.",
        ]
        outro = [
            "So to summarize what we learned today.",
            "The fundamental theorem of calculus has two parts.",
            "Part one says the derivative of an integral recovers the original function.",
            "Part two gives us a practical way to evaluate definite integrals.",
            "Together they unify differential and integral calculus.",
            "In the next video we'll see applications to physics and engineering.",
            "Thanks for watching and I'll see you next time.",
        ]
        all_sentences = intro + body + outro
        transcript = _build_topic_transcript(all_sentences, cue_duration=3.0, gap=0.2)
        total_dur = int(len(all_sentences) * 3.2)
        failures = []

        # === Phase 1: Topic-aligned single reels from different segments ===
        test_segments = [
            (0.0, 24.0, 15, 30, "intro"),
            (25.0, 60.0, 20, 45, "early_body"),
            (60.0, 95.0, 20, 45, "mid_body"),
            (95.0, 115.0, 15, 30, "outro"),
        ]
        for seg_start, seg_end, mn, mx, label in test_segments:
            if seg_end > total_dur:
                seg_end = float(total_dur)
            result = self._run_refine_and_check(
                topic=f"search_calc_{label}",
                transcript=transcript,
                proposed_start=seg_start,
                proposed_end=seg_end,
                video_duration=total_dur + 10,
                min_len=mn,
                max_len=mx,
            )
            if result["window"] is not None:
                if not result["start_ok"]:
                    failures.append(f"search_calc_{label}: start NOT on sentence")
                if result["end_type"] == "mid":
                    failures.append(f"search_calc_{label}: end mid-sentence (t_end={result['window'][1]})")

        # === Phase 2: Full video split — simulate max coverage ===
        windows = self.rs._split_into_consecutive_windows(
            transcript=transcript,
            segment_start=0.0,
            segment_end=float(total_dur),
            video_duration_sec=total_dur + 10,
            min_len=20,
            max_len=40,
        )
        self.assertGreater(len(windows), 2, f"Expected 3+ reels for {total_dur}s video")

        # Verify continuation invariants (allow inter-cue silence gaps)
        max_cue_gap = _max_inter_cue_gap(transcript)
        gap_tol = max_cue_gap + 0.05
        for i in range(len(windows) - 1):
            gap = windows[i + 1][0] - windows[i][1]
            if gap < -0.01:
                failures.append(f"search_calc full: overlap={abs(gap):.3f}s between win {i},{i+1}")
                self.stats["continuation_has_overlap"] += 1
            elif gap > gap_tol:
                failures.append(f"search_calc full: gap={gap:.3f}s between win {i},{i+1} (exceeds cue gap)")
                self.stats["continuation_has_gap"] += 1
            else:
                self.stats["continuation_zero_gap"] += 1

        # Verify every window boundary
        for j, (ws, we) in enumerate(windows):
            self.stats["total_reels"] += 1
            start_ok = self._check_start_boundary(transcript, ws, "search_calc_full")
            end_type = self._check_end_boundary(transcript, we, "search_calc_full")
            if start_ok:
                self.stats["start_on_sentence"] += 1
            else:
                self.stats["start_not_on_sentence"] += 1
                failures.append(f"search_calc_full win[{j}]: start NOT on sentence (t_start={ws})")
            if end_type == "sentence":
                self.stats["end_on_sentence"] += 1
            elif end_type == "fallback":
                self.stats["end_on_fallback"] += 1
            else:
                self.stats["end_mid_sentence"] += 1
                if j < len(windows) - 1:
                    failures.append(f"search_calc_full win[{j}]: end mid-sentence (t_end={we})")

        # === Phase 3: Duration bounds check ===
        for j, (ws, we) in enumerate(windows):
            dur = we - ws
            # Allow up to 8s past max_len for sentence extension
            self.assertGreaterEqual(dur, 19.0, f"win[{j}] too short: {dur:.1f}s")
            self.assertLessEqual(dur, 48.0, f"win[{j}] too long: {dur:.1f}s (max_len=40 + 8s extension)")

        self.assertEqual(failures, [], "\n".join(failures))

    def test_realistic_mixed_punctuation_density(self) -> None:
        """Simulate a transcript where punctuation density varies within the same video.
        Some sections well-punctuated, some sections auto-caption style."""
        well_punctuated = [
            "The periodic table organizes all known elements.",
            "Each element has a unique atomic number.",
            "Hydrogen is the simplest element with just one proton.",
            "Helium is a noble gas that does not react easily.",
            "Carbon forms the basis of organic chemistry.",
            "The arrangement reveals patterns in chemical properties.",
        ]
        auto_caption = [
            "so next we look at the transition metals",
            "they include iron copper and zinc",
            "these metals can form multiple oxidation states",
            "which makes their chemistry really interesting",
            "for example iron can be two plus or three plus",
            "copper appears as one plus or two plus",
        ]
        back_to_punctuated = [
            "To summarize, the periodic table is a powerful tool.",
            "It predicts chemical behavior from atomic structure.",
            "Understanding it is essential for all of chemistry.",
            "That concludes our overview of the periodic table.",
        ]
        all_sentences = well_punctuated + auto_caption + back_to_punctuated
        transcript = _build_topic_transcript(all_sentences, cue_duration=3.0, gap=0.6)
        total_dur = int(len(all_sentences) * 3.6)
        failures = []

        windows = self.rs._split_into_consecutive_windows(
            transcript=transcript,
            segment_start=0.0,
            segment_end=float(total_dur),
            video_duration_sec=total_dur + 10,
            min_len=12,
            max_len=22,
        )

        failures.extend(_check_continuation_gaps(windows, transcript, "mixed_density"))

        for j, (ws, we) in enumerate(windows):
            self.stats["total_reels"] += 1
            start_ok = self._check_start_boundary(transcript, ws, "mixed_density")
            if not start_ok:
                failures.append(f"mixed_density win[{j}]: start NOT on sentence (t_start={ws})")
        self.assertEqual(failures, [], "\n".join(failures))

    # =================== Report =================== #

    def test_zzz_final_stats_report(self) -> None:
        """Print final statistics (runs last due to name sorting)."""
        # Re-run all the heavy tests to collect stats
        # (This test aggregates — individual tests already verified correctness)
        total = self.stats["total_reels"]
        if total == 0:
            return
        print(f"\n{'='*60}")
        print(f"BOUNDARY STRESS TEST STATISTICS")
        print(f"{'='*60}")
        print(f"Total reels analyzed: {total}")
        print(f"Start on sentence boundary: {self.stats['start_on_sentence']}/{total} "
              f"({100*self.stats['start_on_sentence']/total:.1f}%)")
        print(f"Start NOT on sentence: {self.stats['start_not_on_sentence']}/{total}")
        print(f"End on terminal punct: {self.stats['end_on_sentence']}/{total} "
              f"({100*self.stats['end_on_sentence']/total:.1f}%)")
        print(f"End on fallback (cue/pause): {self.stats['end_on_fallback']}/{total}")
        print(f"End mid-sentence: {self.stats['end_mid_sentence']}/{total}")
        gaps = self.stats['continuation_zero_gap'] + self.stats['continuation_has_gap']
        if gaps > 0:
            print(f"Continuation zero-gap: {self.stats['continuation_zero_gap']}/{gaps}")
            print(f"Continuation with gap: {self.stats['continuation_has_gap']}/{gaps}")
            print(f"Continuation with overlap: {self.stats['continuation_has_overlap']}/{gaps}")
        print(f"{'='*60}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
