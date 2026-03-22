"""
eval_local.py — 本地评估脚本
用法：
    python eval_local.py \
        --questions QA_pairs/questions.txt \
        --answers   QA_pairs/answers_gt.txt \
        --preds     QA_pairs/predictions.txt

输入格式：三个 txt 文件，每行一条，行数必须完全对齐。
输出：Exact Match、F1、以及每道题的明细（方便找错误案例）。

F1/EM 计算逻辑与 professor 的 evaluate-v1.1.py 完全一致。
支持多参考答案：answers.txt 里用 | 分隔，如 "Soda Hall|Soda Hall 387"
"""

import argparse
import re
import string
from collections import Counter


# ─── 与官方脚本完全相同的 normalize + 评分函数 ────────────────────────────────

def normalize_answer(s: str) -> str:
    """小写、去标点、去冠词、合并空格。"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens   = normalize_answer(ground_truth).split()
    common      = Counter(pred_tokens) & Counter(gt_tokens)
    num_same    = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def best_score(metric_fn, prediction: str, ground_truths: list) -> float:
    """多参考答案时取最高分。"""
    return max(metric_fn(prediction, gt) for gt in ground_truths)


# ─── 读文件 ────────────────────────────────────────────────────────────────────

def read_lines(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ─── 主逻辑 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="本地 F1/EM 评估")
    parser.add_argument("--questions", default="QA_pairs/questions.txt",
                        help="问题文件（每行一条）")
    parser.add_argument("--answers",   default="QA_pairs/answers_gt.txt",
                        help="参考答案文件（每行一条，多答案用 | 分隔）")
    parser.add_argument("--preds",     default="QA_pairs/predictions.txt",
                        help="模型预测文件（每行一条）")
    parser.add_argument("--show-errors", action="store_true",
                        help="只打印 F1=0 的失败案例")
    args = parser.parse_args()

    questions   = read_lines(args.questions)
    answers_raw = read_lines(args.answers)
    predictions = read_lines(args.preds)

    # 行数对齐检查
    assert len(questions) == len(answers_raw) == len(predictions), (
        f"行数不一致！questions={len(questions)}, "
        f"answers={len(answers_raw)}, preds={len(predictions)}"
    )

    total = len(questions)
    total_f1 = 0.0
    total_em = 0.0
    results  = []

    for i, (q, ans_raw, pred) in enumerate(zip(questions, answers_raw, predictions)):
        # 支持多参考答案，用 | 分隔
        ground_truths = [a.strip() for a in ans_raw.split("|") if a.strip()]

        em = best_score(exact_match_score, pred, ground_truths)
        f1 = best_score(f1_score,          pred, ground_truths)

        total_em += float(em)
        total_f1 += f1
        results.append({
            "idx":   i + 1,
            "question": q,
            "answer":   ans_raw,
            "pred":     pred,
            "em":       em,
            "f1":       round(f1, 4),
        })

    em_pct = 100.0 * total_em / total
    f1_pct = 100.0 * total_f1 / total

    # ── 汇总输出 ──────────────────────────────────────────────────────────────
    print("\n" + "="*52)
    print(f"  评估结果（共 {total} 题）")
    print("="*52)
    print(f"  Exact Match : {em_pct:.2f}%")
    print(f"  F1          : {f1_pct:.2f}%")
    print("="*52)

    # ── 失败案例明细 ──────────────────────────────────────────────────────────
    failures = [r for r in results if r["f1"] == 0.0]
    print(f"\n  F1=0 的失败案例：{len(failures)} / {total} 题")

    if args.show_errors or len(failures) <= -1:
        print()
        for r in failures:
            print(f"  [{r['idx']:>3}] Q : {r['question']}")
            print(f"        A : {r['answer']}")
            print(f"       预测: {r['pred']}")
            print()
    else:
        print(f"  （加 --show-errors 参数可打印所有失败案例）\n")

    # ── 部分正确案例（0 < F1 < 1）──────────────────────────────────────────────
    partial = [r for r in results if 0.0 < r["f1"] < 1.0]
    print(f"  部分正确（0 < F1 < 1）：{len(partial)} / {total} 题")
    print(f"  完全正确（EM=1）       ：{int(total_em)} / {total} 题")
    print()


if __name__ == "__main__":
    main()