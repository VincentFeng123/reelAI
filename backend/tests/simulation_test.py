"""
Comprehensive simulation test for the ReelAI pipeline.
Tests a wide range of topics from ultra-niche to ultra-broad.
Evaluates: relevance, educational quality, clip boundary precision, continuity.
"""

import json
import re
import sys
import time
import requests
from dataclasses import dataclass, field
from youtube_transcript_api import YouTubeTranscriptApi

API_BASE = "https://reelai-production.up.railway.app"
OWNER_KEY = "testsimulationkey00100000000000001"
HEADERS_JSON = {
    "Content-Type": "application/json",
    "x-studyreels-owner-key": OWNER_KEY,
}

# ---------------------------------------------------------------------------
# Test topics: ultra-niche -> niche -> moderate -> broad -> ultra-broad
# ---------------------------------------------------------------------------
TEST_TOPICS = [
    # ULTRA-NICHE
    {"subject": "Krebs Cycle", "text": "The Krebs cycle, also known as the citric acid cycle, is a series of chemical reactions in the mitochondrial matrix that oxidizes acetyl-CoA to CO2 and generates NADH, FADH2, and GTP. It is a central metabolic pathway connecting carbohydrate, fat, and protein metabolism.", "tier": "ultra-niche"},
    {"subject": "Taylor Series Convergence", "text": "A Taylor series is an infinite sum of terms calculated from the values of a function's derivatives at a single point. The radius of convergence determines where the series converges. The ratio test and root test are common methods to determine convergence.", "tier": "ultra-niche"},
    # NICHE
    {"subject": "CRISPR Gene Editing", "text": "CRISPR-Cas9 is a genome editing tool that uses a guide RNA to direct the Cas9 enzyme to a specific DNA sequence. It creates a double-strand break allowing for gene knockout, insertion, or correction. CRISPR has applications in medicine, agriculture, and basic research.", "tier": "niche"},
    {"subject": "Fourier Transform", "text": "The Fourier transform decomposes a function of time into the frequencies that make it up. It transforms a signal from the time domain to the frequency domain. Applications include signal processing, image analysis, and solving differential equations.", "tier": "niche"},
    # MODERATE
    {"subject": "Mitosis", "text": "Mitosis is a type of cell division that produces two genetically identical daughter cells from a single parent cell. The stages are prophase, metaphase, anaphase, and telophase. It is essential for growth, repair, and asexual reproduction in organisms.", "tier": "moderate"},
    {"subject": "Quantum Entanglement", "text": "Quantum entanglement is a phenomenon where two particles become correlated such that the quantum state of one instantly influences the other, regardless of distance. Einstein called it spooky action at a distance. It is fundamental to quantum computing and quantum cryptography.", "tier": "moderate"},
    {"subject": "Game Theory", "text": "Game theory is the study of mathematical models of strategic interaction among rational agents. Key concepts include Nash equilibrium, dominant strategies, and the prisoner's dilemma. It applies to economics, political science, and evolutionary biology.", "tier": "moderate"},
    # BROAD
    {"subject": "World War 2", "text": "World War 2 was a global conflict from 1939 to 1945 involving the Allied and Axis powers. Key events include the invasion of Poland, the Battle of Stalingrad, D-Day, and the atomic bombings of Hiroshima and Nagasaki. It resulted in an estimated 70-85 million deaths.", "tier": "broad"},
    {"subject": "Climate Change", "text": "Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities, especially burning fossil fuels, have been the main driver since the industrial revolution. Effects include rising sea levels, extreme weather events, and biodiversity loss.", "tier": "broad"},
    {"subject": "Machine Learning", "text": "Machine learning is a subset of artificial intelligence where systems learn from data to improve performance on tasks without being explicitly programmed. Types include supervised learning, unsupervised learning, and reinforcement learning. Applications span image recognition, natural language processing, and autonomous vehicles.", "tier": "broad"},
    # ULTRA-BROAD
    {"subject": "Biology", "text": "Biology is the scientific study of life and living organisms. It covers topics from molecular biology and genetics to ecology and evolution. Key concepts include cell theory, natural selection, homeostasis, and the central dogma of molecular biology.", "tier": "ultra-broad"},
    {"subject": "Mathematics", "text": "Mathematics is the abstract science of number, quantity, and space. Major branches include algebra, calculus, geometry, and statistics. Mathematical reasoning underpins physics, engineering, computer science, and economics.", "tier": "ultra-broad"},
]


@dataclass
class ClipEvaluation:
    video_id: str
    video_title: str
    t_start: float
    t_end: float
    clip_duration: float
    video_duration: int
    # Transcript analysis
    transcript_found: bool = False
    cue_count: int = 0
    clip_text: str = ""
    first_cue_text: str = ""
    last_cue_text: str = ""
    # Boundary quality
    starts_clean: bool = False  # starts at sentence/topic boundary
    ends_clean: bool = False    # ends at sentence/topic boundary
    starts_at_intro: bool = False  # starts at topic introduction
    boundary_score: float = 0.0  # 0-1
    # Relevance
    on_topic: bool = False
    educational: bool = False
    relevance_notes: str = ""
    # Raw data
    query_strategy: str = ""
    relevance_score: float = 0.0
    matched_terms: list = field(default_factory=list)


@dataclass
class TopicResult:
    subject: str
    tier: str
    material_id: str = ""
    concept_id: str = ""
    reels_count: int = 0
    clips: list = field(default_factory=list)
    error: str = ""
    generation_time_sec: float = 0.0


def extract_video_id(url: str) -> str:
    m = re.search(r'embed/([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else ""


def is_sentence_start(text: str) -> bool:
    """Check if text begins at a sentence boundary."""
    if not text.strip():
        return False
    first_char = text.strip()[0]
    # Starts with uppercase, number, or quote = likely sentence start
    if first_char.isupper() or first_char.isdigit() or first_char in '"\'(':
        return True
    # Common sentence starters
    starters = ['so ', 'now ', 'the ', 'this ', 'here ', 'let ', 'and ', 'but ', 'in ', 'we ']
    lower = text.strip().lower()
    # If starts with lowercase but is a common conjunction after a period, still OK
    return False


def is_sentence_end(text: str) -> bool:
    """Check if text ends at a sentence boundary."""
    stripped = text.rstrip()
    if not stripped:
        return False
    return stripped[-1] in '.!?'


def is_topic_intro(text: str, subject: str) -> bool:
    """Check if text appears to introduce the topic."""
    lower = text.lower()[:200]
    subject_lower = subject.lower()
    intro_patterns = [
        f"what is {subject_lower}",
        f"{subject_lower} is ",
        f"today we",
        f"in this video",
        f"let's talk about",
        f"we're going to",
        f"let me explain",
        f"welcome",
        f"introduction to",
        f"let's learn",
    ]
    return any(p in lower for p in intro_patterns)


def assess_educational_quality(
    title: str, description: str, text: str,
    channel_name: str = "", video_duration_sec: int = 0, ai_summary: str = "",
) -> tuple[bool, str]:
    """Assess if the video is genuinely educational using all available metadata."""
    title_lower = title.lower()
    desc_lower = description.lower()
    combined = f"{title_lower} {desc_lower}"

    # Red flags for non-educational content
    red_flags = ['reaction', 'vlog', 'unboxing', 'prank', 'challenge', 'asmr',
                 'gameplay', 'lets play', 'mukbang', 'haul', 'grwm', 'top 10',
                 'ranking every', 'tier list']
    for flag in red_flags:
        if flag in title_lower:
            return False, f"Red flag: '{flag}' in title"

    edu_signals = 0.0
    reasons = []

    # Signal 1: Known educational channel (most authoritative signal)
    # Matches the production KNOWN_EDUCATIONAL_CHANNELS set in reels.py
    known_edu = [
        '3blue1brown', 'amoeba sisters', 'bozeman science', 'brilliant',
        'bright side of mathematics', 'cgp grey', 'computerphile', 'crash course',
        'domain of science', 'dr. becky', 'dr. trefor bazett', 'engineerguy',
        'fireship', 'freecodecamp', 'jbstatistics', 'khan academy', 'kurzgesagt',
        'lumen learning', 'mark rober', 'mathologer', 'minutephysics',
        'mit opencourseware', 'nancypi', 'national geographic', 'numberphile',
        'organic chemistry tutor', 'patrickjmt', 'pbs space time', 'physics girl',
        'professor dave explains', 'professor leonard', 'real engineering',
        'science click', 'scishow', 'sixty symbols', 'smarter every day',
        'stand-up maths', 'steve mould', 'ted-ed', 'the coding train',
        'the engineer guy', 'the organic chemistry tutor', 'tibees', 'tom scott',
        'two minute papers', 'veritasium', 'vsauce', 'zach star',
        # Additional well-known educational channels
        'mit', 'ted', 'simplilearn', 'freeschool', 'pbs', 'pbs eons',
    ]
    channel_lower = channel_name.lower().strip()
    if channel_lower and any(ch in channel_lower for ch in known_edu):
        edu_signals += 1.5
        reasons.append(f"known educational channel: {channel_name}")
    elif any(ch in combined for ch in known_edu):
        edu_signals += 1.0
        reasons.append("known educational channel (in metadata)")

    # Channel name contains educational keywords (e.g., "Physics Videos", "Geo History")
    if channel_lower and not any(ch in channel_lower for ch in known_edu):
        edu_channel_words = ['science', 'physics', 'history', 'biology', 'chemistry',
                             'math', 'education', 'professor', 'academy', 'lecture',
                             'engineering', 'medical', 'learning', 'tutorial',
                             'university', 'institute', 'geographic']
        ch_hits = sum(1 for w in edu_channel_words if w in channel_lower)
        if ch_hits >= 1:
            edu_signals += 0.5
            reasons.append(f"educational channel name ({channel_name})")

    # Signal 2: Title keywords
    title_edu_words = ['explained', 'tutorial', 'lecture', 'lesson', 'course', 'learn',
                       'introduction', 'intro', 'how', 'what is', 'guide', 'basics',
                       'chapter', 'part', 'fundamentals', 'overview', 'crash course',
                       'made simple', 'made easy', 'for beginners', '101', 'animated',
                       'animation', 'documentary', 'science', 'history', 'math',
                       'biology', 'chemistry', 'physics', 'calculus', 'algebra',
                       'geometry', 'proof', 'theorem', 'derivation', 'equation']
    title_hits = sum(1 for w in title_edu_words if w in title_lower)
    if title_hits >= 1:
        edu_signals += 0.5 + 0.2 * min(3, title_hits)
        reasons.append(f"educational keywords in title ({title_hits})")

    # Signal 3: Description keywords
    desc_edu_words = ['learn', 'education', 'course', 'academy', 'university',
                      'professor', 'subscribe', 'khan', 'mit', 'ted', 'lecture',
                      'patreon', 'support', 'free', 'practice', 'textbook',
                      'curriculum', 'syllabus', 'exam', 'quiz', 'homework',
                      'chapter', 'lesson', 'student', 'teaching']
    desc_hits = sum(1 for w in desc_edu_words if w in desc_lower)
    if desc_hits >= 1:
        edu_signals += 0.3 + 0.15 * min(3, desc_hits)
        reasons.append(f"educational keywords in description ({desc_hits})")

    # Signal 4: Description structure
    if re.search(r'\d{1,2}:\d{2}', description):
        edu_signals += 0.5
        reasons.append("has timestamps in description")
    if len(description) > 500:
        edu_signals += 0.6
        reasons.append("very detailed description")
    elif len(description) > 200:
        edu_signals += 0.4
        reasons.append("detailed description")
    if 'http' in desc_lower:
        edu_signals += 0.2
        reasons.append("has links")

    # Signal 5: Title structure patterns
    if re.search(r'\b(part|chapter|lecture|episode|module|unit)\s+\d', title_lower):
        edu_signals += 0.5
        reasons.append("series structure in title")
    if re.search(r'^\d+\.?\s+', title_lower):
        edu_signals += 0.3
        reasons.append("numbered title")

    # Signal 6: Duration band (educational videos tend to be 5-30 min)
    if 300 <= video_duration_sec <= 1800:
        edu_signals += 0.3
        reasons.append(f"educational duration band ({video_duration_sec}s)")
    elif 180 <= video_duration_sec <= 3600:
        edu_signals += 0.15
        reasons.append(f"acceptable duration ({video_duration_sec}s)")

    # Signal 7: AI summary (server already assessed educational value)
    if ai_summary and len(ai_summary) > 40:
        edu_signals += 0.3
        reasons.append("has AI summary")

    # Signal 8: Transcript vocabulary (when available)
    if text:
        words = text.split()
        if len(words) > 50:
            unique_ratio = len(set(w.lower() for w in words)) / max(1, len(words))
            if unique_ratio > 0.35:
                edu_signals += 0.5
                reasons.append(f"vocabulary diversity: {unique_ratio:.2f}")
            edu_signals += 0.3
            reasons.append(f"substantive transcript ({len(words)} words)")

    is_edu = edu_signals >= 1.0
    return is_edu, "; ".join(reasons) if reasons else "no educational signals"


def assess_topic_relevance(clip_text: str, subject: str, keywords: list) -> tuple[bool, str]:
    """Check if clip text is actually about the subject."""
    if not clip_text:
        return False, "no transcript"

    text_lower = clip_text.lower()
    subject_words = subject.lower().split()

    # Check for subject mention
    subject_hits = sum(1 for w in subject_words if w in text_lower)
    subject_coverage = subject_hits / max(1, len(subject_words))

    # Check for keyword mentions
    keyword_hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    keyword_coverage = keyword_hits / max(1, len(keywords)) if keywords else 0

    is_relevant = subject_coverage >= 0.5 or keyword_coverage >= 0.3
    notes = f"subject coverage: {subject_coverage:.0%}, keyword coverage: {keyword_coverage:.0%}"

    return is_relevant, notes


def evaluate_clip(reel: dict, subject: str, keywords: list) -> ClipEvaluation:
    """Full evaluation of a single clip."""
    video_url = reel.get("video_url", "")
    video_id = extract_video_id(video_url)
    t_start = reel.get("t_start", 0)
    t_end = reel.get("t_end", 0)

    ev = ClipEvaluation(
        video_id=video_id,
        video_title=reel.get("video_title", ""),
        t_start=t_start,
        t_end=t_end,
        clip_duration=t_end - t_start,
        video_duration=reel.get("video_duration_sec", 0),
        query_strategy=reel.get("query_strategy", ""),
        relevance_score=reel.get("relevance_score", 0),
        matched_terms=reel.get("matched_terms", []),
    )

    if not video_id:
        ev.relevance_notes = "no video ID"
        return ev

    # Try to fetch the transcript. If IP-blocked, fall back to backend-provided
    # captions/snippet for boundary analysis.
    try:
        for attempt in range(2):
            try:
                transcript = YouTubeTranscriptApi().fetch(video_id, languages=['en']).to_raw_data()
                break
            except Exception as retry_err:
                if "IpBlocked" in type(retry_err).__name__ and attempt < 1:
                    time.sleep(3)
                    continue
                # Fall back to backend captions
                transcript = None
                break

        if transcript:
            clip_cues = [c for c in transcript if c['start'] + c['duration'] > t_start and c['start'] < t_end]
        else:
            # Use backend-provided captions as fallback
            captions = reel.get("captions", [])
            snippet = reel.get("transcript_snippet", "")
            if captions and isinstance(captions, list) and captions[0].get("text"):
                clip_cues = captions
            elif snippet:
                clip_cues = [{"start": t_start, "end": t_end, "text": snippet}]
            else:
                clip_cues = []

        if clip_cues:
            ev.transcript_found = True
            ev.cue_count = len(clip_cues)
            ev.clip_text = ' '.join(str(c.get('text', '')) for c in clip_cues)
            ev.first_cue_text = str(clip_cues[0].get('text', ''))
            ev.last_cue_text = str(clip_cues[-1].get('text', ''))

            # Boundary analysis
            ev.starts_clean = is_sentence_start(ev.clip_text)
            ev.ends_clean = is_sentence_end(ev.clip_text)
            ev.starts_at_intro = is_topic_intro(ev.clip_text, subject)

            # Boundary score: 0.0-1.0
            score = 0.0
            if ev.starts_clean:
                score += 0.3
            if ev.starts_at_intro:
                score += 0.2
            if ev.ends_clean:
                score += 0.5  # ending clean is more important
            ev.boundary_score = score

            # Relevance check
            ev.on_topic, ev.relevance_notes = assess_topic_relevance(ev.clip_text, subject, keywords)

            # Educational check
            ev.educational, edu_notes = assess_educational_quality(
                reel.get("video_title", ""),
                reel.get("video_description", ""),
                ev.clip_text,
                channel_name=reel.get("channel_name", ""),
                video_duration_sec=reel.get("video_duration_sec", 0),
                ai_summary=reel.get("ai_summary", ""),
            )
            ev.relevance_notes += f" | edu: {edu_notes}"
    except Exception as e:
        ev.relevance_notes = f"transcript fetch failed: {e}"

    return ev


def create_material(subject: str, text: str) -> tuple[str, str]:
    """Create a material and return (material_id, first_concept_id)."""
    resp = requests.post(
        f"{API_BASE}/api/material",
        headers={"x-studyreels-owner-key": OWNER_KEY},
        data={"subject_tag": subject, "text": text},
        timeout=30,
    )
    data = resp.json()
    material_id = data.get("material_id", "")
    concepts = data.get("extracted_concepts", [])
    # Pick the first concept, or the one matching subject best
    concept_id = ""
    if concepts:
        # Try to find concept closest to subject
        subject_lower = subject.lower()
        best = concepts[0]
        for c in concepts:
            if subject_lower in c.get("title", "").lower():
                best = c
                break
        concept_id = best.get("id", "")
    return material_id, concept_id


def generate_reels(material_id: str, concept_id: str, num: int = 3) -> dict:
    """Generate reels and return the response."""
    resp = requests.post(
        f"{API_BASE}/api/reels/generate",
        headers=HEADERS_JSON,
        json={
            "material_id": material_id,
            "concept_id": concept_id,
            "num_reels": num,
            "generation_mode": "fast",
            "target_clip_duration_sec": 60,
            "target_clip_duration_min_sec": 15,
            "target_clip_duration_max_sec": 120,
        },
        timeout=120,
    )
    return resp.json()


def run_topic_test(topic: dict) -> TopicResult:
    """Run a full test for one topic."""
    result = TopicResult(subject=topic["subject"], tier=topic["tier"])

    try:
        # Step 1: Create material
        material_id, concept_id = create_material(topic["subject"], topic["text"])
        result.material_id = material_id
        result.concept_id = concept_id

        if not material_id or not concept_id:
            result.error = "failed to create material or extract concepts"
            return result

        # Step 2: Generate reels
        t0 = time.time()
        gen_data = generate_reels(material_id, concept_id, num=3)
        result.generation_time_sec = time.time() - t0

        reels = gen_data.get("reels", [])
        result.reels_count = len(reels)

        if not reels:
            result.error = f"no reels generated (response: {json.dumps(gen_data)[:200]})"
            return result

        # Step 3: Evaluate each clip
        keywords = topic["text"].lower().split()[:20]
        for reel in reels:
            ev = evaluate_clip(reel, topic["subject"], keywords)
            result.clips.append(ev)
            time.sleep(1.5)  # be nice to YouTube — avoid IP blocking

    except Exception as e:
        result.error = str(e)

    return result


def print_report(results: list[TopicResult]):
    """Print a comprehensive report."""
    total_clips = 0
    total_on_topic = 0
    total_educational = 0
    total_clean_start = 0
    total_clean_end = 0
    total_intro_start = 0
    total_with_transcript = 0
    boundary_scores = []

    print("\n" + "=" * 100)
    print("COMPREHENSIVE PIPELINE SIMULATION REPORT")
    print("=" * 100)

    for tier in ["ultra-niche", "niche", "moderate", "broad", "ultra-broad"]:
        tier_results = [r for r in results if r.tier == tier]
        if not tier_results:
            continue

        print(f"\n{'─' * 100}")
        print(f"  TIER: {tier.upper()}")
        print(f"{'─' * 100}")

        for result in tier_results:
            print(f"\n  [{result.subject}] — {result.reels_count} reels, {result.generation_time_sec:.1f}s generation")
            if result.error:
                print(f"    ERROR: {result.error}")
                continue

            for i, clip in enumerate(result.clips):
                total_clips += 1
                if clip.transcript_found:
                    total_with_transcript += 1
                if clip.on_topic:
                    total_on_topic += 1
                if clip.educational:
                    total_educational += 1
                if clip.starts_clean:
                    total_clean_start += 1
                if clip.ends_clean:
                    total_clean_end += 1
                if clip.starts_at_intro:
                    total_intro_start += 1
                boundary_scores.append(clip.boundary_score)

                start_icon = "✅" if clip.starts_clean else "❌"
                end_icon = "✅" if clip.ends_clean else "❌"
                topic_icon = "✅" if clip.on_topic else "❌"
                edu_icon = "✅" if clip.educational else "⚠️"

                print(f"\n    Reel {i+1}: {clip.video_title[:70]}")
                print(f"      Video: {clip.video_id} | Clip: {clip.t_start:.1f}s - {clip.t_end:.1f}s ({clip.clip_duration:.0f}s of {clip.video_duration}s)")
                print(f"      Strategy: {clip.query_strategy} | Relevance: {clip.relevance_score:.3f}")
                print(f"      On-topic: {topic_icon} | Educational: {edu_icon} | Start: {start_icon} | End: {end_icon} | Boundary: {clip.boundary_score:.1f}")

                if clip.transcript_found:
                    print(f"      FIRST: \"{clip.clip_text[:100]}...\"")
                    print(f"      LAST:  \"...{clip.clip_text[-100:]}\"")
                else:
                    print(f"      (no transcript available)")

                if clip.relevance_notes:
                    print(f"      Notes: {clip.relevance_notes[:150]}")

    # Summary
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    print(f"  Total topics tested:    {len(results)}")
    print(f"  Total clips generated:  {total_clips}")
    print(f"  Transcripts fetched:    {total_with_transcript}/{total_clips}")
    print()
    print(f"  RELEVANCE")
    print(f"    On-topic:             {total_on_topic}/{total_with_transcript} ({100*total_on_topic/max(1,total_with_transcript):.0f}%)")
    print(f"    Educational:          {total_educational}/{total_with_transcript} ({100*total_educational/max(1,total_with_transcript):.0f}%)")
    print()
    print(f"  CLIP BOUNDARIES")
    print(f"    Clean starts:         {total_clean_start}/{total_with_transcript} ({100*total_clean_start/max(1,total_with_transcript):.0f}%)")
    print(f"    Starts at intro:      {total_intro_start}/{total_with_transcript} ({100*total_intro_start/max(1,total_with_transcript):.0f}%)")
    print(f"    Clean ends:           {total_clean_end}/{total_with_transcript} ({100*total_clean_end/max(1,total_with_transcript):.0f}%)")
    print(f"    Avg boundary score:   {sum(boundary_scores)/max(1,len(boundary_scores)):.2f}/1.00")
    print()

    # Per-tier breakdown
    print(f"  PER-TIER BREAKDOWN")
    for tier in ["ultra-niche", "niche", "moderate", "broad", "ultra-broad"]:
        tier_clips = []
        for r in results:
            if r.tier == tier:
                tier_clips.extend(r.clips)
        if not tier_clips:
            continue
        t_clips = [c for c in tier_clips if c.transcript_found]
        on_t = sum(1 for c in t_clips if c.on_topic)
        edu = sum(1 for c in t_clips if c.educational)
        clean_s = sum(1 for c in t_clips if c.starts_clean)
        clean_e = sum(1 for c in t_clips if c.ends_clean)
        n = max(1, len(t_clips))
        print(f"    {tier:15s}: {len(tier_clips):2d} clips | on-topic {100*on_t/n:.0f}% | edu {100*edu/n:.0f}% | clean-start {100*clean_s/n:.0f}% | clean-end {100*clean_e/n:.0f}%")

    print(f"\n{'=' * 100}")

    # Failure modes
    failures = [(r.subject, r.error) for r in results if r.error]
    if failures:
        print("\n  FAILURES:")
        for subj, err in failures:
            print(f"    [{subj}]: {err[:200]}")

    bad_ends = [(c.video_title[:50], c.clip_text[-80:]) for r in results for c in r.clips if c.transcript_found and not c.ends_clean]
    if bad_ends:
        print(f"\n  ABRUPT ENDINGS ({len(bad_ends)} clips):")
        for title, ending in bad_ends[:10]:
            print(f"    {title}... => \"...{ending}\"")


if __name__ == "__main__":
    print("Starting comprehensive pipeline simulation...")
    print(f"Testing {len(TEST_TOPICS)} topics across 5 tiers\n")

    results = []
    for i, topic in enumerate(TEST_TOPICS):
        print(f"[{i+1}/{len(TEST_TOPICS)}] Testing: {topic['subject']} ({topic['tier']})...", end=" ", flush=True)
        result = run_topic_test(topic)
        results.append(result)
        status = f"{result.reels_count} reels" if not result.error else f"ERROR: {result.error[:60]}"
        print(status)
        time.sleep(0.5)  # rate limit courtesy

    print_report(results)
