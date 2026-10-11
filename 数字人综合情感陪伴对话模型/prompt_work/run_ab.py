#!/usr/bin/env python3
"""Paired prompt A/B evaluation against a single already running local vLLM service."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.message_builder import conversation_text
from core.output_schema import EMOTIONS, PROFILE_LABELS, extract_object, normalize_prediction, parse_prediction
from copy import deepcopy
from core.prompt_loader import load_system_prompt, prompt_metadata
from training.create_validation_split import choose_dev_conversations
from training.evaluate_lora import sentence_bleu4, rouge_l_f1, classification_diagnostics
sys.path.insert(0, str(ROOT / "participant"))
from participant.run_inference import request_chat_completion, load_config, percentile95
from style_postprocessor import StylePostprocessor

DEFAULT_DATA = ROOT / "训练-验证-数据集/val/val_public.jsonl"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dataset(path, split, seed=42):
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    counts = {r["conversation_id"]: len(r["prediction_points"]) for r in rows}
    if len(counts) != len(rows):
        raise ValueError("duplicate conversation_id")
    selected = choose_dev_conversations(counts, seed, .70)
    records = []
    for row in rows:
        cid = row["conversation_id"]
        if (cid in selected) != (split == "dev"):
            continue
        turns = sorted(row["turns"], key=lambda x: x["turn_id"])
        by_id = {t["turn_id"]: t for t in turns}
        if len(by_id) != len(turns):
            raise ValueError("duplicate turn_id")
        for i, point in enumerate(row["prediction_points"]):
            target_id, response_id = point["target_user_turn_id"], point["assistant_response_turn_id"]
            if by_id[target_id]["role"] != "user" or by_id[response_id]["role"] != "assistant":
                raise ValueError("invalid prediction point roles")
            target = point["target"]
            if target["response_text"] != by_id[response_id]["content"]:
                raise ValueError("target / turn mismatch")
            history = [{"role":t["role"],"content":t["content"]} for t in turns
                       if t["turn_id"] <= target_id and t["turn_id"] != response_id]
            if not history or history[-1]["role"] != "user" or history[-1]["content"] != by_id[target_id]["content"]:
                raise ValueError("future leak or malformed target")
            records.append({"id":f"{cid}_p{i}", "conversation_id":cid, "history":history, "target":target})
    if not records:
        raise ValueError("empty split")
    return records

def selected_rows(rows, tier, seed):
    if tier == "full":
        return rows
    n = 8 if tier == "smoke" else 64
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), min(n, len(rows))))
    return [rows[i] for i in indices]

def validity(raw):
    obj = isinstance(raw, dict)
    profile = raw.get("user_profile") if obj else None
    profile = profile if isinstance(profile, dict) else {}
    result = {
        "json_object": obj,
        "exact_fields": obj and set(raw) == {"response_text","emotion_label","user_profile","memory_refs"},
        "response_nonempty": obj and isinstance(raw.get("response_text"),str) and bool(raw["response_text"].strip()),
        "emotion_valid": obj and raw.get("emotion_label") in EMOTIONS if obj and isinstance(raw.get("emotion_label"),str) else False,
        "memory_valid": obj and isinstance(raw.get("memory_refs"),list) and len(raw["memory_refs"])==len(set(str(x) for x in raw["memory_refs"])) and all(isinstance(x,str) and __import__("re").fullmatch(r"mem_[0-9]{6}",x) for x in raw["memory_refs"]),
    }
    for key, labels in PROFILE_LABELS.items():
        value = profile.get(key)
        result["profile_"+key+"_valid"] = isinstance(value,list) and all(isinstance(x,str) and x in labels for x in value) and len(value)==len(set(str(x) for x in value))
    result["profile_exact_fields"] = set(profile)==set(PROFILE_LABELS)
    return result

def metrics(rows):
    size = len(rows)
    correct = [r["prediction"]["emotion_label"] == r["target"]["emotion_label"] for r in rows]
    predicted = [r["prediction"]["emotion_label"] for r in rows]
    actual = [r["target"]["emotion_label"] for r in rows]
    matrix = {t:{p:0 for p in sorted(EMOTIONS)} for t in sorted(EMOTIONS)}
    for t,p in zip(actual,predicted):
        if p in EMOTIONS:
            matrix[t][p] += 1
    classes = {}
    for label in sorted(EMOTIONS):
        tp = sum(p==label and t==label for p,t in zip(predicted,actual))
        fp = sum(p==label and t!=label for p,t in zip(predicted,actual))
        fn = sum(p!=label and t==label for p,t in zip(predicted,actual))
        prec = tp/(tp+fp) if tp+fp else 0
        rec = tp/(tp+fn) if tp+fn else 0
        classes[label] = {"precision":prec,"recall":rec,"f1":2*prec*rec/(prec+rec) if prec+rec else 0,"support":sum(t==label for t in actual)}
    profiles = {key:sum(set(r["prediction"]["user_profile"][key])==set(r["target"]["user_profile"][key]) for r in rows)/size for key in PROFILE_LABELS}
    latency = [r["latency_ms"] for r in rows if r["error"] is None]
    flags = set().union(*(r["validity"].keys() for r in rows))
    return {
        "samples":size,"inference_failures":sum(r["error"] is not None for r in rows),
        "json_parse_failures":sum(r.get("parse_error") is not None for r in rows),
        "emotion_accuracy":sum(correct)/size,
        "emotion_macro_f1":sum(v["f1"] for v in classes.values())/len(classes),
        "emotion_classes":classes,"confusion_matrix":matrix,
        "bleu_4":sum(sentence_bleu4(r["target"]["response_text"],r["prediction"]["response_text"]) for r in rows)/size,
        "rouge_l":sum(rouge_l_f1(r["target"]["response_text"],r["prediction"]["response_text"]) for r in rows)/size,
        "profile_exact_set_accuracy":profiles,"profile_mean_accuracy":sum(profiles.values())/len(profiles),
        "raw_validity_rates":{key:sum(bool(r["validity"].get(key)) for r in rows)/size for key in sorted(flags)},
        "average_latency_ms":statistics.fmean(latency) if latency else None,
        "p95_latency_ms":percentile95(latency) if latency else None,
        "bertscore":None,"bertscore_status":"not computed (optional external model download prohibited by default)",
    }

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts",nargs="+",required=True)
    parser.add_argument("--tier",choices=("smoke","fast","full"),default="smoke")
    parser.add_argument("--split",choices=("dev","holdout"),default="dev")
    parser.add_argument("--data-file",type=Path,default=DEFAULT_DATA)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"prompt_work"/"runs")
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    if len(args.prompts)<2 or len(set(args.prompts))!=len(args.prompts):
        parser.error("specify at least two distinct prompts")
    if args.split=="holdout" and args.tier!="full":
        parser.error("holdout can only be run as full; reserve for final confirmation")
    config=load_config()
    if config["backend"]!="openai_compatible":
        parser.error("A/B uses one prestarted OpenAI-compatible service; set EVALUATION_BACKEND=openai_compatible")
    config_public={key:value for key,value in config.items() if key not in ("api_key","api_options")}
    config_public["api_options_redacted"]=bool(config.get("api_options"))
    processor=StylePostprocessor()
    rows=selected_rows(dataset(args.data_file,args.split,args.seed),args.tier,args.seed)
    manifest_path=ROOT/"training"/"validation_split_manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest_hash=manifest.get("raw_file_hash")
    raw_hash=digest(args.data_file)
    import uuid
    run_dir=args.output_dir/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+args.tier+"_"+uuid.uuid4().hex[:6])
    run_dir.mkdir(parents=True,exist_ok=False)
    summaries={}
    all_results={}
    for name in args.prompts:
        prompt=load_system_prompt(name)
        meta=prompt_metadata(name)
        key=meta["prompt_version"]
        if key in summaries:
            raise ValueError("duplicate resolved prompt version")
        results=[]
        out_dir=run_dir/key
        out_dir.mkdir()
        (out_dir/"prompt.txt").write_text(prompt,encoding="utf-8")
        for row in rows:
            messages=[{"role":"system","content":prompt},{"role":"user","content":conversation_text(row["history"],row["id"])}]
            start=time.perf_counter()
            raw_text=""
            error=None
            parse_error=None
            json_valid=False
            try:
                raw_text=request_chat_completion(config,messages,config["generation"])
                try:
                    raw=extract_object(raw_text)
                    json_valid=True
                except ValueError as exc:
                    parse_error=str(exc)
                    raw={}
            except Exception as exc:
                error=f"{type(exc).__name__}: {str(exc)[:200]}"
                raw={}
            elapsed=(time.perf_counter()-start)*1000
            normalized=parse_prediction(row["id"],raw_text)[0] if error is None else normalize_prediction(row["id"],{})
            final_prediction=deepcopy(normalized)
            if error is None and parse_error is None:
                final_prediction=processor.apply(normalized,row["history"])
            results.append({"id":row["id"],"raw_text":raw_text,"raw_object":raw,
                            "validity":{**validity(raw),"json_object":json_valid},"prediction":normalized,"postprocessed_prediction":final_prediction,
                            "target":row["target"],"error":error,"parse_error":parse_error,"latency_ms":elapsed})
            print(f"{key} {len(results)}/{len(rows)} {row['id']}",flush=True)
        with (out_dir/"predictions.jsonl").open("w",encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r,ensure_ascii=False)+"\n")
        scores=metrics(results)
        final_rows=[{**r,"prediction":r["postprocessed_prediction"]} for r in results]
        final_scores=metrics(final_rows)
        summaries[key]={"prompt":meta,"metrics":scores,"postprocessed_metrics":final_scores}
        all_results[key]=results
        (out_dir/"metrics.json").write_text(json.dumps({"model_raw_normalized":scores,"after_classifier":final_scores},ensure_ascii=False,indent=2),encoding="utf-8")
    keys=list(summaries)
    a,b=all_results[keys[0]],all_results[keys[1]]
    changes=[]
    for x,y in zip(a,b):
        if x["id"]!=y["id"]:
            raise ValueError("A/B alignment mismatch")
        ca=x["prediction"]["emotion_label"]==x["target"]["emotion_label"]
        cb=y["prediction"]["emotion_label"]==y["target"]["emotion_label"]
        changes.append({"id":x["id"],"outcome":"improved" if cb and not ca else "regressed" if ca and not cb else "unchanged",
                        "baseline_emotion":x["prediction"]["emotion_label"],"candidate_emotion":y["prediction"]["emotion_label"],
                        "gold_emotion":x["target"]["emotion_label"]})
    improvements=sum(c["outcome"]=="improved" for c in changes)
    regressions=sum(c["outcome"]=="regressed" for c in changes)
    diff=summaries[keys[1]]["metrics"]["emotion_macro_f1"]-summaries[keys[0]]["metrics"]["emotion_macro_f1"]
    recommendation="REVIEW (proxy only; human reply review required)"
    report={"status":"local proxy only, not official score","data_sha256":raw_hash,"manifest_raw_sha256":manifest_hash,"manifest_sha256_matches":raw_hash==manifest_hash if manifest_hash else None,
            "tier":args.tier,"split":args.split,"seed":args.seed,"config":config_public,
            "versions":summaries,"paired_emotion_improvements":improvements,
            "paired_emotion_regressions":regressions,"macro_f1_delta":diff,"recommendation":recommendation}
    (run_dir/"comparison.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (run_dir/"paired_cases.jsonl").write_text("".join(json.dumps(c,ensure_ascii=False)+"\n" for c in changes),encoding="utf-8")
    (run_dir/"README.md").write_text("# Human review\n\nReview representative responses in each predictions.jsonl for empathy, relevance, boundaries and naturalness. Do not use holdout for iterative tuning.\n",encoding="utf-8")
    print(json.dumps({"report":str(run_dir/"comparison.json"),"improved":improvements,"regressed":regressions},ensure_ascii=False))

if __name__=="__main__":
    main()
