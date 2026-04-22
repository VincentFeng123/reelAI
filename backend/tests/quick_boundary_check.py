#!/usr/bin/env python3
"""
Quick boundary check — run this when YouTube is accessible.
Usage: python tests/quick_boundary_check.py
"""
import sys, re, json
sys.path.insert(0, ".")
from youtube_transcript_api import YouTubeTranscriptApi
from app.services.reels import ReelService
from app.services.embeddings import EmbeddingService
from app.services.youtube import YouTubeService

VIDEOS = [
    ("WUvTyaaNkzM", "3B1B Calculus", ["calculus"], 1150),
    ("aircAruvnKk", "3B1B Neural Nets", ["neural networks"], 1150),
    ("IHZwWFHWa-w", "3B1B Gradient Descent", ["gradient descent"], 1260),
    ("rfscVS0vtbw", "fCC Python (auto)", ["Python"], 14400),
    ("d4EgbgTm0Bg", "3B1B Quaternions", ["quaternions"], 1800),
]

rs = ReelService(embedding_service=EmbeddingService(), youtube_service=YouTubeService())
api = YouTubeTranscriptApi()
total_r = total_s = total_e = total_ch = 0

for vid_id, title, terms, dur in VIDEOS:
    try:
        raw = api.fetch(vid_id, languages=["en"])
        cues = [{"start": c.start, "duration": c.duration, "text": c.text} for c in raw]
    except Exception as e:
        print(f"  SKIP {title}: {type(e).__name__}")
        continue
    is_p = rs._transcript_has_terminal_punct(
        [{"start":c["start"],"end":c["start"]+c.get("duration",0),"text":c["text"]} for c in cues])
    segs = rs._topic_cut_segments_for_concept(
        transcript=cues, video_id=vid_id, video_duration_sec=dur,
        clip_min_len=20, clip_max_len=55, max_segments=6, concept_terms=terms)
    if not segs: continue
    ch={}; le=None; reels=[]
    for seg in sorted(segs, key=lambda s:(s.t_start,getattr(s,"cluster_sub_index",0))):
        sp=seg.t_end-seg.t_start; cg=str(getattr(seg,"cluster_group_id","") or "")
        pv=ch.get(cg) if cg else None
        ef=float(pv) if pv is not None else (float(le) if le and abs(seg.t_start-le)<=2.0 else float(seg.t_start))
        if sp>71:
            w=rs._split_into_consecutive_windows(transcript=cues,segment_start=ef,segment_end=seg.t_end,video_duration_sec=dur,min_len=20,max_len=55)
        else:
            rm=int(max(sp+16,55)); rn=max(1,min(20,int(max(1,sp*0.6))))
            s=rs._refine_clip_window_from_transcript(transcript=cues,proposed_start=ef,proposed_end=seg.t_end,video_duration_sec=dur,min_len=rn,max_len=rm,min_start=ef)
            w=[s] if s else []
        for wi in w:
            if wi: reels.append((wi,cg))
        lw=[x for x in w if x]
        if lw: le=float(lw[-1][1]); (ch.update({cg:le}) if cg else None)
    if not reels: continue
    vs=ve=0
    for ri,(win,cg) in enumerate(reels):
        total_r+=1
        si=ei=None
        for i,c in enumerate(cues):
            if abs(c["start"]-win[0])<0.5: si=i; break
        for i,c in enumerate(cues):
            if abs((c["start"]+c.get("duration",0))-win[1])<0.5: ei=i; break
        sok=True
        if si and si>0:
            pt=cues[si-1]["text"].strip(); ct=cues[si]["text"].strip()
            sok=(bool(pt) and pt[-1] in ".!?…") or not pt or (bool(ct) and ct[0].isupper() and bool(re.search(r"[.!?…]",pt))) or not is_p
        eok=True
        if ei is not None and is_p:
            et=cues[ei]["text"].strip(); eok=bool(et) and et[-1] in ".!?…"
        if not sok: vs+=1; total_s+=1
        if not eok: ve+=1; total_e+=1
    cgg={}
    for ri,(w,cg) in enumerate(reels):
        if cg: cgg.setdefault(cg,[]).append(ri)
    cok=True
    for gid,idx in cgg.items():
        if len(idx)>=2:
            for i in range(len(idx)-1):
                if idx[i+1]!=idx[i]+1: cok=False; total_ch+=1
    tag="punct" if is_p else "AUTO"
    ic="✓" if vs==0 and ve==0 and cok else "✗"
    print(f"  {ic} {title:35s} {len(reels):3d}r {tag:5s} s={vs} e={ve}")

print(f"\nTotal: {total_r}r  start_bad={total_s}  end_bad={total_e}  chain_breaks={total_ch}")
