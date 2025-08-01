import os
import pandas as pd

DATA_DIR = "data"
SYMPTOM_CSV = os.path.join(DATA_DIR, "disease_symptom.csv")


def normalize(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


class SymptomChecker:
    def __init__(self):
        if not os.path.exists(SYMPTOM_CSV):
            raise FileNotFoundError(f"❌ File not found: {SYMPTOM_CSV}")
        self.df = pd.read_csv(SYMPTOM_CSV)

        self.symptom_cols = [col for col in self.df.columns if col.lower() != "disease"]
        self.df["symptom_tokens"] = self.df[self.symptom_cols].apply(
            lambda row: set(col for col in self.symptom_cols if row[col] == 1),
            axis=1
        )
        self.df["Disease"] = self.df["Disease"].astype(str)

    def predict(self, symptom_string: str, top_n: int = 5, min_score: float = 0.0) -> pd.DataFrame:
        user_tokens = set(normalize(sym) for sym in symptom_string.split(",") if sym)
        if not user_tokens:
            return pd.DataFrame(columns=self.df.columns)

        def compute_score(disease_tokens):
            intersection = user_tokens & disease_tokens
            union = user_tokens | disease_tokens
            jaccard = len(intersection) / len(union) if union else 0
            coverage = len(intersection) / len(user_tokens) if user_tokens else 0
            precision = len(intersection) / len(disease_tokens) if disease_tokens else 0
            return 0.4 * jaccard + 0.4 * coverage + 0.2 * precision

        self.df["score"] = self.df["symptom_tokens"].apply(compute_score)
        filtered = self.df[self.df["score"] >= min_score]
        
        # Sort by score and get unique diseases (keep the highest score for each disease)
        sorted_df = filtered.sort_values("score", ascending=False)
        unique_diseases = sorted_df.drop_duplicates(subset=["Disease"], keep="first")
        
        return unique_diseases.head(top_n)


_checker = None


def predict_diseases(symptom_string: str, top_n: int = 5, min_score: float = 0.0) -> pd.DataFrame:
    global _checker
    if _checker is None:
        _checker = SymptomChecker()
    return _checker.predict(symptom_string, top_n, min_score)


def format_symptom_response(symptom_string: str, results_df: pd.DataFrame) -> str:
    if results_df.empty:
        return "❌ No matches found."

    user_tokens = set(normalize(sym) for sym in symptom_string.split(",") if sym)

    # Show all exact matches if they exist, otherwise show top 3 unique diseases
    exact_matches = results_df[results_df["score"] == 1.0]
    if not exact_matches.empty:
        results_df = exact_matches
    else:
        # Ensure we get exactly 3 unique diseases for top 3
        results_df = results_df.head(3)

    neat_symptoms = ', '.join([
        s.replace('_', ' ').title() for s in symptom_string.split(',') if s.strip()
    ])
    output = [
        f" 🤖 Possible conditions for: <span style=\"color:#1976d2;font-weight:600;\">{neat_symptoms}</span>"
    ]

    for _, row in results_df.iterrows():
        disease = row["Disease"]
        score = row["score"]
        matched = user_tokens & row["symptom_tokens"]

        # Skip if no symptoms matched (for non-exact matches)
        if score < 1.0 and len(matched) < 1:
            continue

        output.append(
            f"#### 🩺 {disease}\n"
            f"**Match Score:** {int(score * 100)}%"
        )

    return "\n\n---\n\n".join(output) if output else "❌ No matches found."