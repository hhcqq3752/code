import argparse
import json
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm
from datasets import load_from_disk


def load_jsonl(path):
    """Load a JSONL file and return a list of items."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    """Save a list of dicts to a JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Batch-streaming decontamination with detailed progress bars"
    )
    parser.add_argument("--reference_path", type=str, required=True)
    parser.add_argument("--suspect_path", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, default="decontam_matches")
    parser.add_argument("--ngram_min", type=int, default=13)
    parser.add_argument("--ngram_max", type=int, default=13)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=10_000)
    parser.add_argument("--save_clean", action="store_true")
    args = parser.parse_args()

    # 1) Load reference
    print(f"▶ Loading reference from {args.reference_path} …")
    ref_ds = load_from_disk(args.reference_path)
    # with open(args.reference_path, "r") as f:
    #     ref_ds = [json.loads(line) for line in tqdm(f, desc="Loading reference")]
    reference_prompts = [item["chosen"][0]["content"] for item in ref_ds]
    n_ref = len(reference_prompts)
    print(f"✔ {n_ref} reference items loaded")

    # 2) Load suspect
    print(f"▶ Loading suspect from {args.suspect_path} …")
    # sus_ds = load_from_disk(args.suspect_path)
    with open(args.suspect_path, "r") as f:
        sus_ds = [json.loads(line) for line in tqdm(f, desc="Loading suspect")]
    n_sus = len(sus_ds)
    print(f"✔ {n_sus} suspect items loaded")

    # 3) Fit vectorizer
    print("▶ Fitting CountVectorizer on reference prompts …")
    vectorizer = CountVectorizer(
        ngram_range=(args.ngram_min, args.ngram_max),
        binary=True
    ).fit(reference_prompts)
    ref_vec = vectorizer.transform(reference_prompts)  # (n_ref × V)
    print("✔ Vectorizer ready, vocabulary size:", len(vectorizer.vocabulary_))

    # Prepare state
    best_scores = np.zeros(n_ref, dtype=float)
    best_hits   = [None] * n_ref
    contaminated_indices = set()

    # 4) Process in batches with tqdm
    total_batches = (n_sus + args.batch_size - 1) // args.batch_size
    for start in tqdm(
        range(0, n_sus, args.batch_size),
        desc="Batches",
        total=total_batches,
        unit="batch"
    ):
        end = min(start + args.batch_size, n_sus)
        batch_size = end - start

        # 4a) Extract prompts
        batch_prompts = [
            sus_ds[i]["chosen"][0]["content"]
            for i in range(start, end)
        ]

        # 4b) Vectorize this batch
        sus_vec = vectorizer.transform(batch_prompts)  # (batch_size × V)

        # 4c) Compute ref×sus similarities (sparse)
        sim = ref_vec.dot(sus_vec.T).tocoo()

        # 4d) Mark contaminated suspects by column-sum > threshold
        col_sums = np.array(sim.tocsr().sum(axis=0)).ravel()
        for j_batch, col_sum in tqdm(
            enumerate(col_sums),
            total=batch_size,
            desc=f"  Scanning suspects {start}-{end}",
            unit="sus"
        ):
            if col_sum > args.threshold:
                contaminated_indices.add(start + j_batch)

        # 4e) Update per-reference best hits
        for i_ref, j_batch, score in tqdm(
            zip(sim.row, sim.col, sim.data),
            total=sim.nnz,
            desc=f"  Updating best hits in batch {start}-{end}",
            unit="overlap"
        ):
            if score > best_scores[i_ref]:
                best_scores[i_ref] = score
                best_hits[i_ref] = {
                    "reference_index":  int(i_ref),
                    "suspect_index":    int(start + j_batch),
                    "reference_prompt": reference_prompts[i_ref],
                    "suspect_prompt":   batch_prompts[j_batch],
                    "overlap_score":    float(score),
                    "suspect_item":     sus_ds[int(start + j_batch)],
                }

    # 5) Collect + sort
    matches = [
        h for h in best_hits
        if h is not None and h["overlap_score"] > args.threshold
    ]
    matches.sort(key=lambda x: x["overlap_score"], reverse=True)

    # 6) Persist results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_matches = f"{args.output_prefix}_{ts}_matches.jsonl"
    print(f"▶ Saving {len(matches)} matches to {out_matches} …")
    save_jsonl(matches, out_matches)

    print(f"✔ Found {len(contaminated_indices)} contaminated suspect items total")
    if matches:
        low, high = matches[-1]["overlap_score"], matches[0]["overlap_score"]
        print(f"  - Score range: {low:.2f}-{high:.2f}")
        print("  - Top 5 matches:")
        for i, m in enumerate(matches[:5], 1):
            print(f"    {i}. Ref #{m['reference_index']} vs Sus #{m['suspect_index']} (score {m['overlap_score']:.2f})")

    # 7) Optionally save the cleaned dataset
    if args.save_clean:
        clean_idxs = sorted(set(range(n_sus)) - contaminated_indices)
        # clean_ds = sus_ds.select(clean_idxs)
        clean_ds = [sus_ds[i] for i in clean_idxs]
        clean_dir = f"{args.output_prefix}_{ts}_cleaned.jsonl"
        print(f"▶ Saving cleaned suspect ({len(clean_idxs)} items) to {clean_dir} …")
        # clean_ds.save_to_disk(clean_dir)
        with open(clean_dir, "w") as f:
            for item in clean_ds:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print("✔ Clean dataset saved")

if __name__ == "__main__":
    main()
