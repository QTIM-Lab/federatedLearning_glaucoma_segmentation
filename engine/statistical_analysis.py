import os
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Statistical analysis of model performance using Friedman and Wilcoxon tests")
    parser.add_argument("--eval_type", type=str, choices=['disc', 'cup'], default='cup',
                        help="Type of segmentation to analyze: 'disc' or 'cup' (default: cup)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Custom input directory (default: scores/disc or scores/cup)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory (default: Statistics/disc or Statistics/cup)")
    parser.add_argument("--skip-summaries", action="store_true",
                        help="Skip generating optional summary files and plots (only create pairwise_wilcoxon.csv files needed for plotting)")
    return parser.parse_args()

# === Configuration ===
args = parse_args()

# Determine repository root (engine directory is one level up from this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)

eval_type = args.eval_type

# Use custom directories if provided, otherwise use defaults
if args.input_dir:
    input_dir = args.input_dir
else:
    input_dir = os.path.join(repo_root, "scores", eval_type)

if args.output_dir:
    output_dir = args.output_dir
else:
    output_dir = os.path.join(repo_root, "Statistics", eval_type)

os.makedirs(output_dir, exist_ok=True)

print(f"{'='*60}")
print(f"Statistical Analysis for {eval_type.upper()} Segmentation")
print(f"{'='*60}")
print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")
print(f"{'='*60}\n")

# Optionally filter for .csv files only
csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

# To store summaries
friedman_summary = []
global_wins = []
winner_matrix_rows = []
winner_matrix_rows_with_pvalues = []

# === Process each dataset ===
for file in csv_files:
    dataset_name = file.replace(".csv", "")
    print(f"\nProcessing {dataset_name}...")

    df = pd.read_csv(os.path.join(input_dir, file))
    pivot_df = df.pivot(index='image_name', columns='model_name', values='dice_score').dropna()
    
    if pivot_df.shape[1] < 2:
        print(f"Skipping {dataset_name}: fewer than 2 models.")
        continue

    # 1. Friedman Test
    try:
        stat, p_friedman = friedmanchisquare(*[pivot_df[col] for col in pivot_df.columns])
        friedman_summary.append({'Dataset': dataset_name, 'Friedman_Statistic': stat, 'Friedman_p': p_friedman})
    except ValueError:
        print(f"Skipping {dataset_name}: Friedman test failed (insufficient data).")
        continue

    # 2. Wilcoxon signed-rank tests
    model_pairs = list(combinations(pivot_df.columns, 2))
    p_values = []
    test_stats = []
    model_means = pivot_df.mean()
    model_medians = pivot_df.median()
    model_stds = pivot_df.std()

    mean_a = []
    mean_b = []
    median_a = []
    median_b = []
    std_a = []
    std_b = []
    better_models = []
    winner = []

    for model_a, model_b in model_pairs:
        try:
            stat, p = wilcoxon(pivot_df[model_a], pivot_df[model_b])
        except:
            stat, p = np.nan, 1.0

        test_stats.append(stat)
        p_values.append(p)

        m_a = model_means[model_a]
        m_b = model_means[model_b]
        mean_a.append(m_a)
        mean_b.append(m_b)
        
        median_a.append(model_medians[model_a])
        median_b.append(model_medians[model_b])
        std_a.append(model_stds[model_a])
        std_b.append(model_stds[model_b])

        if m_a > m_b:
            better_models.append(model_a)
        else:
            better_models.append(model_b)

    # 3. Bonferroni Correction
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

    # 4. Determine significant winners
    for i, (a, b) in enumerate(model_pairs):
        if pvals_corrected[i] < 0.05:
            if model_means[a] > model_means[b]:
                winner.append(a)
                global_wins.append({'Dataset': dataset_name, 'Winner': a})
            else:
                winner.append(b)
                global_wins.append({'Dataset': dataset_name, 'Winner': b})
        else:
            winner.append("None")

    # 5. Save pairwise results
    results = pd.DataFrame({
        'Model_A': [a for a, b in model_pairs],
        'Model_B': [b for a, b in model_pairs],
        'Mean_A': mean_a,
        'Mean_B': mean_b,
        'Median_A': median_a,
        'Median_B': median_b,
        'Std_A': std_a,
        'Std_B': std_b,
        'Wilcoxon_Stat': test_stats,
        'Raw_p': p_values,
        'Corrected_p': pvals_corrected,
        'Reject_Null': reject,
        'Better_Model': better_models,
        'Significant_Winner': winner
    })

    results_path = os.path.join(output_dir, f"{dataset_name}_{eval_type}_pairwise_wilcoxon.csv")
    results.to_csv(results_path, index=False)

    # Build winner matrix row for this dataset
    pairwise_winner_row = {'Dataset': dataset_name}
    for i, (a, b) in enumerate(model_pairs):
        pairwise_winner_row[f"{a} vs {b}"] = winner[i]
    winner_matrix_rows.append(pairwise_winner_row)

    # Build winner matrix with p-values row for this dataset
    pairwise_winner_row_with_pvalues = {'Dataset': dataset_name}
    for i, (a, b) in enumerate(model_pairs):
        pair_name = f"{a} vs {b}"
        pairwise_winner_row_with_pvalues[f"{pair_name} (winner)"] = winner[i]
        pairwise_winner_row_with_pvalues[f"{pair_name} (p)"] = round(pvals_corrected[i], 5)
    winner_matrix_rows_with_pvalues.append(pairwise_winner_row_with_pvalues)


    # 6. Per-dataset significant wins bar plot (OPTIONAL)
    if not args.skip_summaries:
        per_win_counts = results['Significant_Winner'].value_counts()
        win_df = pd.DataFrame({'Model': per_win_counts.index, 'Significant_Wins': per_win_counts.values})
        win_df = win_df[win_df['Model'] != 'None'].sort_values('Significant_Wins', ascending=False)
        win_df.to_csv(os.path.join(output_dir, f"{dataset_name}_significant_win_counts.csv"), index=False)

        plt.figure(figsize=(10, 5))
        sns.barplot(data=win_df, x='Model', y='Significant_Wins', palette='crest')
        plt.title(f"Model Ranking by Significant Wins ({dataset_name})")
        plt.ylabel("Number of Wins (p < 0.05)")
        plt.xlabel("Model Name")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_significant_win_ranking.png"))
        plt.close()

        # 7. Heatmap of corrected p-values
        models = pivot_df.columns.tolist()
        n_models = len(models)
        pval_matrix = np.ones((n_models, n_models))

        for i, (model_a, model_b) in enumerate(model_pairs):
            idx_a = models.index(model_a)
            idx_b = models.index(model_b)
            pval_matrix[idx_a, idx_b] = pvals_corrected[i]
            pval_matrix[idx_b, idx_a] = pvals_corrected[i]

        pval_df = pd.DataFrame(pval_matrix, index=models, columns=models)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(pval_df, annot=True, fmt=".2g", cmap="coolwarm", cbar_kws={'label': 'Corrected p-value'})
        plt.title(f"Heatmap: {dataset_name} (Bonferroni-corrected p-values)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_heatmap.png"))
        plt.close()

    print(f"Finished {dataset_name}")

# === Optional summaries (can be skipped with --skip-summaries flag)
if not args.skip_summaries:
    # === Save Friedman summary
    friedman_df = pd.DataFrame(friedman_summary)
    friedman_df.to_csv(os.path.join(output_dir, "friedman_summary.csv"), index=False)

    # === Save winner matrix summary CSV
    if winner_matrix_rows:
        winner_matrix_df = pd.DataFrame(winner_matrix_rows)
        winner_matrix_df.to_csv(os.path.join(output_dir, "pairwise_significant_winner_matrix.csv"), index=False)
        print("Winner matrix saved: pairwise_significant_winner_matrix.csv")

    # === Save winner matrix with pvalues summary CSV
    if winner_matrix_rows_with_pvalues:
        winner_matrix_with_pvalues_df = pd.DataFrame(winner_matrix_rows_with_pvalues)
        winner_matrix_with_pvalues_df.to_csv(os.path.join(output_dir, "pairwise_significant_winner_matrix_with_pvalues.csv"), index=False)
        print("Winner matrix saved: pairwise_significant_winner_matrix_with_pvalues.csv")

    # === Global leaderboard
    if global_wins:
        global_df = pd.DataFrame(global_wins)
        leaderboard = global_df['Winner'].value_counts().reset_index()
        leaderboard.columns = ['Model', 'Total_Wins']
        leaderboard = leaderboard.sort_values('Total_Wins', ascending=False)
        leaderboard.to_csv(os.path.join(output_dir, "global_model_leaderboard.csv"), index=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=leaderboard, x='Model', y='Total_Wins', palette='rocket')
        plt.title("🌍 Global Model Ranking by Significant Wins (All Datasets)")
        plt.ylabel("Total Pairwise Wins (p < 0.05)")
        plt.xlabel("Model Name")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "global_model_leaderboard.png"))
        plt.close()

        print("\n🏆 Global model leaderboard saved.")

    # === Global pairwise heatmap matrix
    if winner_matrix_rows:
        all_pairs = [col for col in winner_matrix_df.columns if col != 'Dataset']
        model_wins_matrix = {}

        for pair in all_pairs:
            wins = winner_matrix_df[pair].value_counts()
            for model in wins.index:
                if model != "None":
                    model_wins_matrix.setdefault(model, {})
                    model_wins_matrix[model][pair] = wins[model]

        # Convert nested dict to DataFrame and fill missing values with 0
        pairwise_heatmap_df = pd.DataFrame(model_wins_matrix).fillna(0).astype(int).T
        pairwise_heatmap_df = pairwise_heatmap_df.sort_index()

        heatmap_file = os.path.join(output_dir, "pairwise_significant_winner_frequency_matrix.csv")
        pairwise_heatmap_df.to_csv(heatmap_file)

        # Visualize the frequency heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(pairwise_heatmap_df, annot=True, fmt='d', cmap="YlGnBu", linewidths=0.5)
        plt.title("Frequency of Significant Pairwise Wins (Across Datasets)")
        plt.xlabel("Model A vs Model B")
        plt.ylabel("Winner Model")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pairwise_significant_winner_frequency_heatmap.png"))
        plt.close()

        print("Pairwise winner frequency heatmap saved.")

print(f"\nAll results saved to: {output_dir}")
