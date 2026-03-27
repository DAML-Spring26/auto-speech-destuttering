import pandas as pd
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation

RESULTS_CSV = "outputs/llama_test_results.csv"

transform = Compose([
    ToLowerCase(),
    RemovePunctuation()
])

def fluency_score(text):
    words = text.lower().split()
    score = 0

    score -= sum(1 for i in range(1, len(words)) if words[i] == words[i-1])

    fillers = {"uh", "um", "erm", "ah"}
    score -= sum(1 for w in words if w in fillers)

    for i in range(len(words) - 1):
        if words[i] in {"he", "she", "they", "the", "company"} and words[i+1] == "be":
            score -= 1

    score += len(words) * 0.01

    return score

df = pd.read_csv(RESULTS_CSV)

rows = []

for _, row in df.iterrows():
    original = str(row["transcript"]).strip()
    cleaned = str(row["cleaned"]).strip()
    label = str(row["true_label"]).strip()

    if not cleaned or cleaned == "nan":
        continue

    orig_n = transform(original)
    clean_n = transform(cleaned)

    change = wer(orig_n, clean_n)

    orig_score = fluency_score(original)
    clean_score = fluency_score(cleaned)

    improved = clean_score >= orig_score

    rows.append({
        "label_type": label,
        "original": original,
        "cleaned": cleaned,
        "WER_change": change,
        "orig_score": orig_score,
        "clean_score": clean_score,
        "improved": improved
    })

df_out = pd.DataFrame(rows)

if len(df_out) == 0:
    print("No valid rows found.")
    exit()

print("\n=== LLaMA TEXT CHANGE ===")
print("Avg change from original:", df_out["WER_change"].mean())

print("\n=== CHANGE BY LABEL ===")
print(df_out.groupby("label_type")["WER_change"].mean())


improved_count = df_out["improved"].sum()
total = len(df_out)

print("\n=== FLUENCY IMPROVEMENT ===")
print(f"Improved sentences: {improved_count}/{total} ({improved_count/total:.1%})")

print("\n=== SAMPLE EXAMPLES ===\n")

samples = df_out.sample(min(5, len(df_out)))

for _, row in samples.iterrows():
    print("ORIGINAL :", row["original"])
    print("CLEANED  :", row["cleaned"])
    print("IMPROVED :", row["improved"])
    print("-" * 60)