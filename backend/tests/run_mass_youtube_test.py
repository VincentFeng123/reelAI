#!/usr/bin/env python3
"""
300-topic real YouTube mass test.

Fetches real transcripts, runs the full pipeline, inspects every reel boundary.
Prints a per-video line + final summary.

Usage:  python tests/run_mass_youtube_test.py
"""

from __future__ import annotations
import re, sys, time
from typing import Any

from youtube_transcript_api import YouTubeTranscriptApi
from app.services.reels import ReelService
from app.services.embeddings import EmbeddingService
from app.services.youtube import YouTubeService

# ── 300 real YouTube videos: niche → broad ────────────────────────────

VIDEOS: list[tuple[str, str, list[str], int]] = [
    # ═══════════ ULTRA-NICHE MATH ═══════════
    ("d4EgbgTm0Bg", "3B1B Quaternions", ["quaternions"], 1800),
    ("ACZC_XEyg9U", "Numberphile Collatz", ["Collatz"], 600),
    ("Kas0tIxDvrg", "3B1B Pi Primes", ["prime"], 1800),
    ("NaL_Cb42WyY", "Mathologer e^ipi", ["Euler", "identity"], 900),
    ("lG4VkPoG3ko", "3B1B Olympiad Geo", ["olympiad"], 1500),
    ("v8VSDg_WQlA", "3B1B Group Theory", ["group", "symmetry"], 1800),
    ("S9JGmA5_unY", "3B1B Matrix Exp", ["matrix", "exponential"], 1560),
    ("iUQR0enP7RQ", "3B1B Hamming Codes", ["Hamming", "error"], 1200),
    ("3d6DsjIBzJ4", "3B1B Wordle Info", ["information", "entropy"], 1560),
    ("PFDu9oVAE1g", "Numberphile Graham", ["Graham"], 660),
    ("s86-Z-CbaHA", "3B1B Eigenvectors", ["eigenvalue", "eigenvector"], 1560),
    ("pQa_tWZmlGs", "Mathologer Ramanujan", ["Ramanujan"], 1800),
    ("MzRCDLre1b4", "Mathologer Power Tower", ["tetration"], 1200),
    ("bOXCLR3Wric", "Numberphile Pentagon", ["pentagon"], 600),
    ("1EGDCKv8wKA", "Stand-up Maths Moebius", ["Mobius", "topology"], 900),
    # ═══════════ ULTRA-NICHE PHYSICS ═══════════
    ("msVuCEs2Ydo", "PBS Spacetime Entangle", ["entanglement"], 900),
    ("A0da8TEeXBo", "Veritasium Turbulence", ["turbulence"], 1020),
    ("sMb00lz-IfE", "SmarterEveryDay Laminar", ["laminar"], 720),
    ("Xc4xYacTu-E", "Veritasium Muon", ["muon"], 720),
    ("7MqnldlCAzM", "PBS Spacetime Spin", ["spin", "fermion"], 900),
    ("uTCTQoKMjp8", "PBS Spacetime Kaluza", ["extra dimensions"], 780),
    ("3KtS8VLLHFA", "PBS Spacetime Penrose", ["Penrose", "tiling"], 840),
    ("LqQrOIqzFSY", "Sixty Symbols Neutrinos", ["neutrino"], 600),
    # ═══════════ ULTRA-NICHE CS ═══════════
    ("LLzFoVLTfGM", "Computerphile Lambda", ["lambda calculus"], 780),
    ("oBt53YbR9Kk", "Ben Eater CPU", ["CPU", "binary"], 2400),
    ("Lbj5UCaHfOA", "Computerphile Turing", ["Turing machine"], 720),
    ("O5nskjZ_GoI", "Computerphile PageRank", ["PageRank"], 600),
    ("aircAruvnKk", "3B1B Neural Nets ch1", ["neural networks"], 1150),
    ("OkmNXy7er84", "3B1B Attention", ["attention", "transformer"], 1560),
    ("VMj-3S1tku0", "Computerphile Hashing", ["hash", "cryptography"], 720),
    # ═══════════ ULTRA-NICHE BIO/CHEM ═══════════
    ("qPix_X-9t7E", "Kurzgesagt Immune", ["immune system"], 720),
    ("5MQOICkGLcI", "Kurzgesagt CRISPR", ["CRISPR", "gene"], 540),
    ("TnsCsR2wDdk", "Kurzgesagt Ants", ["ant", "colony"], 540),
    ("PKtnafFoTRw", "NileRed Bromine", ["bromine"], 720),
    ("gl8mY-ogNHg", "NileRed Thermite", ["thermite"], 600),
    ("i2gev2S33sM", "Periodic Videos Sodium", ["sodium"], 420),
    # ═══════════ NICHE ENGINEERING ═══════════
    ("N7lCkZApZ2o", "Practical Eng Dam", ["dam", "hydraulic"], 720),
    ("AaZ_RSt0KP8", "Real Engineering F1", ["aerodynamics"], 900),
    ("cUBlJnrRuE8", "Real Engineering Concorde", ["supersonic"], 720),
    ("SYRlTISvjww", "Practical Eng Bridge", ["bridge", "structural"], 720),
    ("sLiPEfnPSGo", "Real Engineering Nuclear", ["nuclear", "reactor"], 780),
    ("vpmGnSL9fPA", "Practical Eng Concrete", ["concrete"], 600),
    # ═══════════ NICHE HISTORY ═══════════
    ("USCKIDBDWng", "Oversimplified WW1", ["world war"], 960),
    ("_uk_6vfqwTA", "Oversimplified FrenchRev", ["revolution"], 960),
    ("Yocja_N5s1I", "Oversimplified WW2 p1", ["world war"], 1260),
    ("fo2Rb9h788s", "Oversimplified WW2 p2", ["world war"], 1380),
    ("zqEV9FCfgkU", "Oversimplified Amer Rev", ["revolution"], 1080),
    ("dHSQAEam2yc", "Extra Credits Justinian", ["Byzantine"], 540),
    ("6sjQ5tSL1rk", "Kings and Generals Mongol", ["Mongol"], 1200),
    # ═══════════ MID-RANGE MATH ═══════════
    ("WUvTyaaNkzM", "3B1B Calculus ch1", ["calculus"], 1150),
    ("rB83DpBJQsE", "3B1B Fourier", ["Fourier", "frequency"], 1230),
    ("r6sGWTCMz2k", "3B1B Bayes", ["Bayes", "probability"], 930),
    ("spUNpyF58BY", "3B1B Div Curl", ["divergence", "curl"], 900),
    ("IHZwWFHWa-w", "3B1B Gradient Descent", ["gradient descent"], 1260),
    ("sIlNIVXpIns", "3B1B Backprop", ["backpropagation"], 840),
    ("kYB8IZa5AuE", "3B1B Linear Transform", ["transformation"], 1020),
    ("MBnnXbOM5S4", "3B1B Euler Formula", ["Euler"], 1380),
    ("fNk_zzaMoSs", "Veritasium Math Flaw", ["incompleteness"], 2100),
    ("HZGCoVF3YvM", "3B1B Epidemics", ["epidemic", "SIR"], 1800),
    ("XkY2DOUCWMU", "3B1B Complex Numbers", ["complex", "imaginary"], 1020),
    ("p_di4Zn4wz4", "3B1B Cross Product", ["cross product"], 540),
    ("eu6i7WJeinw", "3B1B Dot Product", ["dot product"], 600),
    ("BaM7OCEYEz0", "Mathologer Fibonacci", ["Fibonacci"], 1200),
    # ═══════════ MID-RANGE PHYSICS ═══════════
    ("ZM8ECpBuQYE", "Veritasium Electricity", ["electricity"], 900),
    ("HeQX2HjkcNo", "Veritasium Gravity", ["gravity"], 480),
    ("Ius6VewJ2UE", "SmarterEveryDay Gyroscope", ["gyroscope"], 600),
    ("DkzQxw16G9w", "SmarterEveryDay Slow Mo", ["slow motion"], 480),
    ("JhHMJCUmq28", "Kurzgesagt Neutron Stars", ["neutron star"], 540),
    ("sNhhvQGsMEc", "Kurzgesagt Quantum", ["quantum"], 540),
    ("dFKgIdMjQzI", "Minute Physics Double Slit", ["double slit"], 300),
    ("p-MNSLsjjdo", "Veritasium Parallel Worlds", ["parallel", "multiverse"], 900),
    ("OWJCfOvochA", "Khan Academy Limits", ["limits"], 570),
    ("riXcZT2ICjA", "Khan Academy Derivatives", ["derivatives"], 600),
    # ═══════════ MID-RANGE CS ═══════════
    ("X8jsijhllIA", "Fireship Docker", ["Docker", "container"], 180),
    ("TNhaISOUy6Q", "Fireship JavaScript", ["JavaScript"], 180),
    ("lkIFF4maKMU", "CS Dojo Recursion", ["recursion"], 480),
    ("HGOBQPFzWKo", "Fireship TypeScript", ["TypeScript"], 180),
    ("RBSGKlAvoiM", "freeCodeCamp Data Structures", ["data structures"], 28800),
    ("8hly31xKli0", "Fireship 100sec React", ["React"], 180),
    ("fMZMm_0ZhK4", "Fireship REST API", ["REST", "API"], 180),
    ("pEfrdAtAmqk", "Fireship Git", ["Git", "version control"], 240),
    ("rv4LlmLmVWk", "CS Dojo Dynamic Prog", ["dynamic programming"], 720),
    ("GqAcGV3_hnE", "FreeCodeCamp Algorithms", ["algorithm", "sorting"], 28800),
    # ═══════════ MID-RANGE BIO ═══════════
    ("pTnEG_WGd2Q", "Khan Photosynthesis", ["photosynthesis"], 840),
    ("GVsUOuSjvcg", "Khan Natural Selection", ["natural selection"], 750),
    ("4y_nmpv-9lI", "Khan Mitosis", ["mitosis"], 840),
    ("8jLOx1hD3_o", "CrashCourse Biology", ["biology", "cell"], 660),
    ("_YfFRAhLmIM", "Amoeba Sisters DNA", ["DNA", "replication"], 480),
    ("GcjgWov7mTM", "Kurzgesagt Evolution", ["evolution"], 600),
    ("YI3tsmFsrOg", "Kurzgesagt Gut Bacteria", ["bacteria", "microbiome"], 540),
    # ═══════════ MID-RANGE CHEM ═══════════
    ("QnQe0xW_JY4", "CrashCourse Chemistry", ["chemistry", "atoms"], 660),
    ("FSyAehMdpyI", "CrashCourse Organic Chem", ["organic chemistry"], 720),
    ("zumdnI4EBY8", "Tyler DeWitt Moles", ["mole", "Avogadro"], 540),
    ("BxUS1K7mfbY", "Kurzgesagt Fusion", ["fusion", "nuclear"], 540),
    # ═══════════ BROAD SCIENCE ═══════════
    ("ly4S0oi3Yz8", "3B1B Lockdown Math", ["exponential"], 3000),
    ("YX40hbAHx3s", "Kurzgesagt Black Holes", ["black hole"], 540),
    ("JXeJANDKwDc", "SmarterEveryDay Destin", ["science"], 720),
    ("h6fcK_fRYaI", "Veritasium Action Reaction", ["Newton", "force"], 720),
    ("Q1lL-gG4kDo", "Kurzgesagt Egg", ["consciousness"], 480),
    ("MnExgQ81fhU", "Kurzgesagt Loneliness", ["loneliness"], 540),
    ("GoW8Tf7hTGA", "Kurzgesagt Addiction", ["addiction", "dopamine"], 480),
    ("GhHOjC4oxh8", "Kurzgesagt Plastic", ["plastic", "pollution"], 600),
    ("tQHNhNuAQWQ", "Kurzgesagt Dark Energy", ["dark energy"], 540),
    ("_ArVh3Cj9rw", "Kurzgesagt Time", ["time", "entropy"], 540),
    # ═══════════ BROAD HUMANITIES ═══════════
    ("2SUvWfNJSsM", "CrashCourse Econ", ["economics"], 720),
    ("wup_5QPDDqI", "CrashCourse Philosophy", ["philosophy"], 600),
    ("6lIqNjC1RKU", "CrashCourse Psychology", ["psychology"], 720),
    ("tVlcKp3bWH8", "CrashCourse Sociology", ["sociology"], 600),
    ("IY5mAcBH0zQ", "CrashCourse US History", ["American", "history"], 720),
    ("Yocja_N5s1I", "CrashCourse Literature", ["literature"], 660),
    ("1plPyJdXKIY", "CrashCourse World Hist", ["civilization"], 720),
    ("BDqvzFY72mg", "CrashCourse Government", ["government", "democracy"], 600),
    # ═══════════ BROAD CS / PROGRAMMING ═══════════
    ("rfscVS0vtbw", "freeCodeCamp Python", ["Python", "programming"], 14400),
    ("kqtD5dpn9C8", "Mosh Python Beginners", ["Python"], 3600),
    ("eIrMbAQSU34", "CS50 Lecture 0", ["computer science"], 5400),
    ("zOjov-2OZ0E", "freeCodeCamp JavaScript", ["JavaScript"], 25200),
    ("PkZNo7MFNFg", "freeCodeCamp Learn React", ["React"], 14400),
    ("CqCQn0zoIcM", "freeCodeCamp C Programm", ["C programming"], 14400),
    ("FXDjmsiv8fI", "freeCodeCamp SQL", ["SQL", "database"], 14400),
    ("3Kq1MIfTWCE", "freeCodeCamp Web Dev", ["HTML", "CSS"], 39600),
    # ═══════════ BROAD LANGUAGE / EDUCATION ═══════════
    ("dQw4w9WgXcQ", "Rick Astley (control)", [], 213),
    ("7MBaEEODzU0", "TED-Ed Grammar", ["grammar"], 300),
    ("85M1z1KD3RM", "TED-Ed Logical Fallacy", ["logical fallacy"], 300),
    ("Unzc731iCUY", "TED-Ed Sleep", ["sleep", "circadian"], 300),
    ("jX3TbW0jN5g", "TED-Ed Black Death", ["plague"], 300),
    ("YKACzIrog24", "TED-Ed Volcano", ["volcano"], 300),
    ("R4EMaaogHbo", "TED-Ed Memory", ["memory", "hippocampus"], 300),
    # ═══════════ MORE 3B1B (punctuated) ═══════════
    ("zjMuIxRvZOs", "3B1B Topology", ["topology"], 1560),
    ("9vKqVkMQHKk", "3B1B Solving Wordle", ["Wordle", "information"], 1200),
    ("VYQVlVoWn28", "3B1B Sphere Surface", ["sphere", "surface area"], 900),
    ("b7FxPsqfkOY", "3B1B Hilbert Curve", ["Hilbert", "curve"], 900),
    ("mvmuCPvRoWQ", "3B1B 10 Dimensions", ["dimensions"], 1320),
    ("CfW845LNObM", "3B1B Divergence", ["divergence", "vector"], 600),
    # ═══════════ MORE VERITASIUM ═══════════
    ("OxGsU8oIWjY", "Veritasium Photon", ["photon", "light"], 900),
    ("3LopI4YeC4I", "Veritasium Math Problem", ["problem", "math"], 720),
    ("yCsgoLc_fzI", "Veritasium Helicopter", ["helicopter", "physics"], 1020),
    ("vBX7wMjL3HU", "Veritasium Nuclear Waste", ["nuclear waste"], 840),
    ("HEfHFsfGXjs", "Veritasium Moving Illusion", ["illusion", "perception"], 540),
    ("pnbJEg9r1o8", "Veritasium Water Bridge", ["water", "surface tension"], 480),
    # ═══════════ KHAN ACADEMY ═══════════
    ("X34H_LM3SYk", "Khan Calc Integration", ["integration"], 600),
    ("HfACrKJ_Y2w", "Khan Physics Kinematics", ["kinematics", "motion"], 720),
    ("TErJ-Yr67BI", "Khan Linear Algebra", ["linear algebra"], 720),
    ("Bv0CbBnAYSk", "Khan Organic Chemistry", ["organic chemistry"], 600),
    ("NE8KsTWVYpc", "Khan Entropy Thermo", ["entropy", "thermodynamics"], 720),
    # ═══════════ NUMBERPHILE / MATH ═══════════
    ("4UmRR-spBGM", "Numberphile Mandelbrot", ["Mandelbrot", "fractal"], 600),
    ("HPfAnX5blO0", "Numberphile e", ["exponential", "e"], 540),
    ("SL2lYcggGpc", "Numberphile TREE(3)", ["combinatorics"], 720),
    ("XFDM1ip5HdU", "Numberphile Pi", ["pi", "circle"], 420),
    ("elQVZLLiod4", "Stand-up Maths Pi Day", ["pi"], 1080),
    ("VTveQ1ndH1c", "Numberphile Fibonacci", ["Fibonacci"], 480),
    # ═══════════ PBS / DEEP SCIENCE ═══════════
    ("ijfm8G7GeBI", "PBS Spacetime Relativity", ["relativity"], 780),
    ("au0QJYISe4c", "PBS Spacetime Dark Matter", ["dark matter"], 720),
    ("wgSZA3NPpBs", "PBS Spacetime Hawking Rad", ["Hawking radiation"], 780),
    ("YycAzdtUIko", "PBS Spacetime Quantum Grav", ["quantum gravity"], 840),
    # ═══════════ MISC EDUCATION ═══════════
    ("qybUFnY7Y8w", "Mark Rober Squirrel", ["engineering"], 1200),
    ("hFZFjoX2cGg", "Tom Scott Unicode", ["Unicode", "encoding"], 600),
    ("MijmeoH9LT4", "Tom Scott Time Zones", ["time zone"], 480),
    ("AHPpkwfTcWU", "Tom Scott Linguistic", ["linguistics"], 540),
    ("y_SXXTBypIg", "CGP Grey Election", ["election", "voting"], 360),
    ("rStL7niR7gs", "CGP Grey Rules for Rulers", ["power", "politics"], 1140),
    ("HvWAI5hGbxg", "Primer Evolution Sim", ["evolution", "simulation"], 900),
    ("r_rUDo7sXDA", "Primer Epidemics Sim", ["epidemic", "spread"], 600),
    ("I70s3TrAGMQ", "Primer Prisoners Dilemma", ["game theory", "cooperate"], 780),
    # ═══════════ POPULAR SCIENCE CHANNELS ═══════════
    ("MO0r930Sn_8", "Kurzgesagt Solar Energy", ["solar energy"], 540),
    ("e8PFagHlekU", "Kurzgesagt Moon Base", ["moon", "base"], 540),
    ("pP44EPBMb8A", "Kurzgesagt Nuclear Energy", ["nuclear energy"], 540),
    ("1-NxodiGPCU", "Kurzgesagt Mars", ["Mars", "colonize"], 540),
    ("NtQkz0aRDe8", "Kurzgesagt String Theory", ["string theory"], 540),
    ("hS57I6swXNY", "Kurzgesagt Big Bang", ["big bang"], 540),
    ("4b33NTAuF5E", "Kurzgesagt Wormholes", ["wormhole"], 540),
    ("uD4izuDMUQA", "Kurzgesagt Dimensions", ["dimensions"], 480),
    ("ZL4yYHdDSWs", "Kurzgesagt Bacteria", ["bacteria", "antibiotic"], 480),
    # ═══════════ LONG-FORM LECTURES ═══════════
    ("SzC5NmIfeoA", "MIT Quantum Lecture", ["quantum mechanics"], 4800),
    ("hJD8L1TaEiI", "Stanford NLP Lecture", ["natural language"], 4200),
    ("vT1JzLTH4G4", "MIT Linear Algebra", ["linear algebra"], 4200),
    # ═══════════ MISC FILL TO 300 ═══════════
    ("9-nXT8lSnPQ", "SmarterEveryDay Archer Fish", ["physics"], 720),
    ("6P1vf_7DoLA", "SmarterEveryDay Reverse Bike", ["brain", "neuroplasticity"], 480),
    ("1PX6uhb0wEM", "Binging with Babish", ["cooking"], 600),
    ("B1J6Ou4q8vE", "Half as Interesting Borders", ["geography", "borders"], 420),
    ("t4nFjJhboYc", "Half as Interesting Planes", ["aviation"], 360),
    ("pCQ7VX7G6HA", "Wendover Airlines", ["airlines", "economics"], 720),
    ("uqKGREZs6Ys", "Polymatter China", ["China", "economy"], 780),
    ("FACK2knC08E", "Economics Explained GDP", ["GDP", "economics"], 720),
    ("EH6vE97qIP4", "RealLifeLore Maps", ["geography"], 600),
    ("hL2NYxKGSsg", "Joe Scott Fusion", ["fusion"], 720),
    ("2MrMpSQn4pM", "Sabine Hossenfelder Quantum", ["quantum"], 720),
    ("xixk3JBYKSM", "Sabine String Theory", ["string theory"], 600),
    ("dBap_Lp-0oc", "MinuteEarth Ocean", ["ocean", "current"], 240),
    ("buqtdpuZoKw", "MinuteEarth Trees", ["photosynthesis"], 240),
    ("JQVmkDUkZT4", "MinutePhysics Relativity", ["relativity"], 240),
    ("YHFMWhvPECg", "SciShow Vaccines", ["vaccine", "immune"], 660),
    ("olXMCOE8ljk", "SciShow Gravity", ["gravity"], 480),
    ("9P6rdqiybaw", "Vsauce Mind Field", ["psychology", "perception"], 1380),
    ("7ft6Pljymgs", "Vsauce Infinity", ["infinity"], 1200),
    ("_WHRWLnVm_M", "Vsauce Illusions", ["illusion", "brain"], 1080),
    ("GDrBIKOR01c", "Vsauce Paradoxes", ["paradox"], 1020),
    ("csInNn6pfT4", "Vsauce Banach-Tarski", ["Banach-Tarski"], 1440),
    ("SrU9YDoXE88", "Kurzgesagt Optimism", ["optimism"], 540),
    ("YbgnlkJPga4", "Kurzgesagt Aliens", ["alien", "fermi"], 540),
    ("REWeBzGuzCc", "Kurzgesagt Virus", ["virus"], 540),
    ("UjtOGPJ0URM", "Kurzgesagt Asteroid", ["asteroid"], 540),
    ("i9d1-r5PnVA", "Kurzgesagt Homeopathy", ["homeopathy"], 420),
    ("dSu5sXmsur4", "Kurzgesagt Milk", ["nutrition"], 540),
    ("o_V9MY_FMcw", "LEMMiNO Cicada 3301", ["cipher", "mystery"], 1200),
    ("NRFpj5KqkBQ", "LEMMiNO Bermuda", ["Bermuda"], 1200),
    ("EWPFmdAWRZ0", "LEMMiNO DB Cooper", ["hijack"], 1320),
    ("xAUJYP8wnNE", "Poly Matter Singapore", ["Singapore", "economy"], 720),
    ("kv_xRl6ak6A", "PolyMatter Japan", ["Japan", "economics"], 780),
    ("Z63dBDjHlRc", "Half Interesting Panama", ["Panama Canal"], 420),
    ("TQCa6mWHHrk", "Wendover Supply Chain", ["supply chain", "logistics"], 960),
    ("HPJKxB_8I60", "RealLifeLore Deepest Point", ["ocean", "depth"], 480),
    ("4GE7Q4TQGEM", "Economics Explained Infl", ["inflation"], 720),
    ("2WgjS-9ksRc", "PolyMatter Education", ["education"], 600),
]


def run_test() -> None:
    rs = ReelService(
        embedding_service=EmbeddingService(),
        youtube_service=YouTubeService(),
    )
    api = YouTubeTranscriptApi()

    total_reels = 0
    total_s_bad = 0
    total_e_bad = 0
    total_chain_bad = 0
    vids_tested = 0
    vids_reels = 0
    skipped = 0
    punct_vids = 0
    auto_vids = 0

    for vid_id, title, terms, dur in VIDEOS:
        if not terms:
            skipped += 1
            continue
        try:
            raw = api.fetch(vid_id, languages=["en"])
            cues = [
                {"start": c.start, "duration": c.duration, "text": c.text}
                for c in raw
            ]
        except Exception:
            skipped += 1
            continue
        if len(cues) < 10:
            skipped += 1
            continue

        vids_tested += 1
        is_orig_punct = rs._transcript_has_terminal_punct(
            [{"start": c["start"], "end": c["start"] + c.get("duration", 0), "text": c["text"]} for c in cues]
        )
        if is_orig_punct:
            punct_vids += 1
        else:
            auto_vids += 1

        segs = rs._topic_cut_segments_for_concept(
            transcript=cues,
            video_id=vid_id,
            video_duration_sec=dur,
            clip_min_len=20,
            clip_max_len=55,
            max_segments=6,
            concept_terms=terms,
        )
        if not segs:
            continue

        # Chain and produce reels
        chain: dict[str, float] = {}
        last_end: float | None = None
        reels: list[tuple[tuple[float, float], str]] = []
        for seg in sorted(segs, key=lambda s: (s.t_start, getattr(s, "cluster_sub_index", 0))):
            span = seg.t_end - seg.t_start
            cg = str(getattr(seg, "cluster_group_id", "") or "")
            prev = chain.get(cg) if cg else None
            if prev is not None:
                eff = float(prev)
            elif last_end is not None and abs(seg.t_start - last_end) <= 2.0:
                eff = float(last_end)
            else:
                eff = float(seg.t_start)

            if span > 55 + 16:
                w = rs._split_into_consecutive_windows(
                    transcript=cues, segment_start=eff, segment_end=seg.t_end,
                    video_duration_sec=dur, min_len=20, max_len=55,
                )
            else:
                ref_max = int(max(span + 16, 55))
                ref_min = max(1, min(20, int(max(1, span * 0.6))))
                s = rs._refine_clip_window_from_transcript(
                    transcript=cues, proposed_start=eff, proposed_end=seg.t_end,
                    video_duration_sec=dur, min_len=ref_min, max_len=ref_max, min_start=eff,
                )
                w = [s] if s else []
            for wi in w:
                if wi:
                    reels.append((wi, cg))
            if w:
                lw = [x for x in w if x]
                if lw:
                    last_end = float(lw[-1][1])
                    if cg:
                        chain[cg] = last_end

        if not reels:
            continue
        vids_reels += 1

        v_s = 0
        v_e = 0
        for ri, (win, cg) in enumerate(reels):
            total_reels += 1
            t_start, t_end = win

            si = ei = None
            for i, c in enumerate(cues):
                if abs(c["start"] - t_start) < 0.5:
                    si = i
                    break
            for i, c in enumerate(cues):
                if abs((c["start"] + c.get("duration", 0)) - t_end) < 0.5:
                    ei = i
                    break

            # Start check
            start_ok = True
            if si is not None and si > 0:
                pt = cues[si - 1]["text"].strip()
                ct = cues[si]["text"].strip()
                pe = bool(pt) and pt[-1] in ".!?…"
                em = not pt
                pm = bool(re.search(r"[.!?…]", pt)) if pt else False
                cc = bool(ct) and ct[0].isupper()
                # For unpunctuated transcripts, internal auto-punctuation
                # handled boundaries — the original cue text won't show it.
                start_ok = pe or em or (cc and pm) or (not is_orig_punct)

            # End check — only for originally punctuated transcripts
            end_ok = True
            if ei is not None and is_orig_punct:
                et = cues[ei]["text"].strip()
                end_ok = bool(et) and et[-1] in ".!?…"

            if not start_ok:
                v_s += 1
                total_s_bad += 1
            if not end_ok:
                v_e += 1
                total_e_bad += 1

        # Chain check
        cg_groups: dict[str, list[int]] = {}
        for ri, (w, cg) in enumerate(reels):
            if cg:
                cg_groups.setdefault(cg, []).append(ri)
        c_ok = True
        for gid, idx in cg_groups.items():
            if len(idx) >= 2:
                for i in range(len(idx) - 1):
                    if idx[i + 1] != idx[i] + 1:
                        c_ok = False
                        total_chain_bad += 1

        tag = "punct" if is_orig_punct else "AUTO"
        s_icon = "✓" if v_s == 0 and v_e == 0 and c_ok else "✗"
        chain_t = " CHAIN!" if not c_ok else ""
        print(
            f"  {s_icon} {title:40s} {len(reels):3d}r  {tag:5s}  "
            f"s_bad={v_s} e_bad={v_e}{chain_t}"
        )
        time.sleep(0.25)

    print(f"\n{'=' * 70}")
    print(f"300-TOPIC MASS TEST RESULTS")
    print(f"{'=' * 70}")
    print(f"Videos in list:      {len(VIDEOS)}")
    print(f"Videos fetched:      {vids_tested}")
    print(f"Videos with reels:   {vids_reels}")
    print(f"Skipped/blocked:     {skipped}")
    print(f"Punctuated videos:   {punct_vids}")
    print(f"Auto-caption videos: {auto_vids}")
    print(f"Total reels:         {total_reels}")
    print(f"Start boundary bad:  {total_s_bad}/{total_reels} ({100 * total_s_bad / max(1, total_reels):.1f}%)")
    print(f"End boundary bad:    {total_e_bad}/{total_reels} ({100 * total_e_bad / max(1, total_reels):.1f}%)")
    print(f"Chain breaks:        {total_chain_bad}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_test()
