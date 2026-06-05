import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

"""
check the sample num of each dataset
"""

dataset_samples = {
    "narrativeqa": 200,
    "qasper":200,
    "multifieldqa_en": 150,
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "gov_report": 200,
    "qmsum": 200,
    "multi_news": 200,
    "trec": 200,
    "triviaqa": 200,
    "samsum": 200,
    "passage_retrieval_en": 200,
    "passage_count": 200,
    "lcc": 500,
    "repobench-p": 500,
}
dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--base_path', type=str, default=None, help="Base directory for prediction files")
    return parser.parse_args(args)


def collect_prediction_files(base_path):
    run2files = {}
    for root, _, files in os.walk(base_path):
        for filename in files:
            if not filename.endswith('.jsonl'):
                continue
            if filename.endswith('_tmp'):
                continue
            file_path = os.path.join(root, filename)
            if root not in run2files:
                run2files[root] = []
            run2files[root].append(file_path)
    for run_dir in run2files:
        run2files[run_dir].sort()
    return run2files

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    score_list = []
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        score_list.append(score)
        total_score += score
    return round(100 * total_score / len(predictions), 2),score_list


if __name__ == '__main__':    
    args = parse_args()
    base_paths = [
        "pred_longbench"
    ]
    for base_path in base_paths:
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base path not found: {base_path}")

        run2files = collect_prediction_files(base_path)
        if args.model is not None:
            run2files = {
                run_dir: file_paths
                for run_dir, file_paths in run2files.items()
                if os.path.basename(run_dir).startswith(args.model)
            }

        print(f"Found {len(run2files)} run directories under {base_path}")

        for run_dir in sorted(run2files.keys()):
            scores = {}
            scores_list = {}
            all_files = run2files[run_dir]

            print(f"Processing {run_dir}")
            print("Evaluating on:", [os.path.basename(fp) for fp in all_files])

            for file_path in all_files:
                predictions, answers, lengths = [], [], []
                dataset = os.path.basename(file_path).split('.')[0]
                all_classes = []

                with open(file_path, "r", encoding="utf-8") as f:
                    line_cnt = 0
                    for line in f:
                        line_cnt += 1
                        data = json.loads(line)
                        # predictions.append(data["pred"])
                        answers.append(data["answers"])
                        all_classes = data.get("all_classes", [])
                        if "length" in data:
                            lengths.append(data["length"])
                        if isinstance(data["pred"], dict) and "thinking_content" in data["pred"]:
                            predictions.append(data["pred"]["pred"]) # adapt to thinking model
                        else:
                            predictions.append(data["pred"])

                    target_samples = dataset_samples.get(dataset, 200)
                    if line_cnt != target_samples:
                        print(f"Error: {dataset} has {line_cnt} samples, expected {target_samples}")
                        continue

                if args.e:
                    score = scorer_e(dataset, predictions, answers, lengths, all_classes)
                    scores[dataset] = score
                else:
                    score, score_list = scorer(dataset, predictions, answers, all_classes)
                    scores[dataset] = score
                    scores_list[dataset] = score_list

            out_path = os.path.join(run_dir, "result.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(scores, f, ensure_ascii=False, indent=4)

            if not args.e:
                list_out_path = os.path.join(run_dir, "list_result.json")
                with open(list_out_path, "w", encoding="utf-8") as f:
                    json.dump(scores_list, f, ensure_ascii=False, indent=4)