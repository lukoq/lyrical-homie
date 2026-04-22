import json
from pathlib import Path
from tqdm import tqdm
from lyrical_homie import LyricalEngine

BASE_DIR = Path(__file__).parent
BENCHMARK_FILE = BASE_DIR / "datasets" / "benchmark_dataset_merged.json"
DB_PATH = BASE_DIR.parent / "vector_db"
REPORT_FILE = Path(__file__).parent / "reports" / "benchmark_report.json"


def run_benchmark():
    engine = LyricalEngine(DB_PATH)

    with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    results = []
    metrics = {
        "intent_correct": 0,
        "retrieval_strict_hit": 0,
        "retrieval_soft_hit": 0,
        "total": len(test_cases)
    }

    print("Lunching benchmark...\n")

    for case in tqdm(test_cases):
        q = case['query']
        expected_id = case['test_id']
        expected_intent = case['expected_intent']
        expected_text = case['expected_child']

        actual_intent = engine.classify_intent(q)
        target_tag = engine.extract_search_tag(q)
        hyde_text = engine.generate_hyde_answer(q, actual_intent, target_tag)

        top_candidates = engine.get_context(q, hyde_text, actual_intent, target_tag)


        # 1. intention
        intent_ok = (actual_intent.upper() == expected_intent.upper())
        if intent_ok: metrics["intent_correct"] += 1

        # 2. strict
        retrieval_ids = [c.get('id') for c in top_candidates]
        strict_hit = expected_id in retrieval_ids

        # 2. soft
        soft_hit = any(expected_text.lower() in c.get('parent', '').lower() for c in top_candidates)
        if strict_hit:
            metrics["retrieval_strict_hit"] += 1
            metrics["retrieval_soft_hit"] += 1
        elif soft_hit:
            metrics["retrieval_soft_hit"] += 1

        results.append({
            "query": q,
            "expected_id": expected_id,
            "intent": {"expected": expected_intent, "actual": actual_intent, "ok": intent_ok},
            "retrieval": {
                "strict_hit": strict_hit,
                "soft_hit": soft_hit,
                "found_ids": retrieval_ids,
                "top_1_found": retrieval_ids[0] if retrieval_ids else None
            },
        })

    print("\n" + "=" * 30)
    print("SCORE:")
    print(f"Intentions (Intent Accuracy):  {(metrics['intent_correct'] / metrics['total']) * 100:.1f}%")
    print(f"Strict Hit Rate (ID Match): {(metrics['retrieval_strict_hit'] / metrics['total']) * 100:.1f}%")
    print(f"Soft Hit Rate (Context):    {(metrics['retrieval_soft_hit'] / metrics['total']) * 100:.1f}%")
    print("=" * 30)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"metrics": metrics, "details": results}, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:  {REPORT_FILE}")


if __name__ == "__main__":
    run_benchmark()