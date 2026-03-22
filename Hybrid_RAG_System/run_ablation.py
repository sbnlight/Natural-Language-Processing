#!/usr/bin/env python3
"""
run_ablation.py — Automated ablation studies script
Used to dynamically modify global configurations in rag.py, run different experimental groups, 
and automatically calculate F1 scores and response times, finally exporting a white-background PNG.
"""

import time
import re
import string
import collections
from pathlib import Path
import pandas as pd  
import matplotlib.pyplot as plt

# Dynamically import your RAG module
import rag 

# ─── Configure paths ───
QUESTIONS_FILE = Path("QA_pairs/questions.txt")
GROUND_TRUTH_FILE = Path("QA_pairs/answers_gt.txt")  

def normalize_answer(s: str) -> str:
    """Standard string cleaning function for official evaluation (Lower, remove punctuation, remove articles)"""
    s = str(s).lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return " ".join(s.split())

def compute_f1_and_em(prediction: str, ground_truth: str) -> tuple:
    """Calculate Token-level F1 and Exact Match (EM) for a single prediction"""
    pred_tokens = normalize_answer(prediction).split()
    
    best_f1 = 0.0
    best_em = 0.0
    
    # Support multiple reference answers (separated by '|')
    for gt in ground_truth.split('|'):
        gt_tokens = normalize_answer(gt).split()
        
        # Exact Match
        if pred_tokens == gt_tokens:
            best_em = 1.0
            
        # F1 Score
        common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
        num_same = sum(common.values())
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            f1 = int(pred_tokens == gt_tokens)
        elif num_same == 0:
            f1 = 0.0
        else:
            precision = 1.0 * num_same / len(pred_tokens)
            recall = 1.0 * num_same / len(gt_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            
        best_f1 = max(best_f1, f1)
        
    return best_f1, best_em

def run_experiment(name: str, overrides: dict, questions: list, ground_truths: list):
    """Run a single experiment group"""
    print(f"\n{'='*50}\nRunning Ablation: {name}\n{'='*50}")
    
    # 1. Dynamically overwrite global variables in rag.py
    for key, value in overrides.items():
        setattr(rag, key, value)
        print(f"  [Config] rag.{key} = {value}")
        
    # 2. Re-instantiate the model (will reload based on new settings like USE_RERANKER)
    model = rag.RAGModel()
    
    # 3. Run inference and time it
    start_time = time.perf_counter()
    predictions = model.predict(questions)
    latency = time.perf_counter() - start_time
    
    # 4. Calculate metrics
    total_f1, total_em = 0.0, 0.0
    for pred, gt in zip(predictions, ground_truths):
        f1, em = compute_f1_and_em(pred, gt)
        total_f1 += f1
        total_em += em
        
    avg_f1 = (total_f1 / len(questions)) * 100
    avg_em = (total_em / len(questions)) * 100
    avg_time_per_q = latency / len(questions)
    
    print(f"  [Result] F1: {avg_f1:.2f}% | EM: {avg_em:.2f}% | Total Time: {latency:.2f}s")
    
    return {
        "Experiment": name,
        "F1 Score (%)": round(avg_f1, 2),
        "Exact Match (%)": round(avg_em, 2),
        "Total Time (s)": round(latency, 2),
        "Latency/Q (s)": round(avg_time_per_q, 3)
    }

def save_table_as_image(df, filename="ablation_results.png"):
    """Render the pandas DataFrame as a PNG image with a white background using Matplotlib"""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    ax.axis('tight')
    
    # Create the table
    table = ax.table(cellText=df.values, 
                     colLabels=df.columns, 
                     loc='center', 
                     cellLoc='center')
    
    # Styling the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Bold the header and give it a light gray background
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')
            
    # Save as PNG with white facecolor
    plt.savefig(filename, facecolor='white', bbox_inches='tight', dpi=300)
    print(f"\nSuccessfully saved white-background table image to: {filename}")

def main():
    if not QUESTIONS_FILE.exists() or not GROUND_TRUTH_FILE.exists():
        print(f"Error: Make sure both {QUESTIONS_FILE} and {GROUND_TRUTH_FILE} exist!")
        print("You need a Ground Truth file (one answer per line) to calculate F1 offline.")
        return

    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        questions =[line.strip() for line in f if line.strip()]
        
    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        ground_truths =[line.strip() for line in f if line.strip()]

    if len(questions) != len(ground_truths):
        print(f"Error: Mismatch in lines! {len(questions)} questions vs {len(ground_truths)} answers.")
        return

    # Core: Define the list of ablation experiments you want to do here!
    experiments =[
        {
            "name": "K=8 + Reranker (Optimal)",
            "overrides": {"FINAL_TOP_K": 8, "USE_RERANKER": True, "USE_DENSE": True}
        },
        {
            "name": "K=4 + Reranker",
            "overrides": {"FINAL_TOP_K": 4, "USE_RERANKER": True, "USE_DENSE": True}
        },
        {
            "name": "K=2 + Reranker",
            "overrides": {"FINAL_TOP_K": 2, "USE_RERANKER": True, "USE_DENSE": True}
        },
        {
            "name": "No Reranker (RRF Only)",
            "overrides": {"FINAL_TOP_K": 8, "USE_RERANKER": False, "USE_DENSE": True}
        },
        {
            "name": "BM25 Only",
            "overrides": {"FINAL_TOP_K": 8, "USE_RERANKER": False, "USE_DENSE": False}
        }
    ]

    results =[]
    
    # Iterate through and run all experiments
    for exp in experiments:
        res = run_experiment(exp["name"], exp["overrides"], questions, ground_truths)
        results.append(res)
        
    # Use Pandas to print a beautiful Markdown table in the terminal
    print("\n\n" + " ABLATION STUDIES RESULTS ".center(60, "━"))
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Generate the white-background PNG image
    save_table_as_image(df, filename="ablation_results.png")
    
    # Also print the Markdown code
    print("\nMarkdown Table for your Written Report (Q4):")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()