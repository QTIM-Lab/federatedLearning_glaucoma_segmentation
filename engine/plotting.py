#!/usr/bin/env python3



"""

Consolidated Plot Generation Toolkit

This script combines all plotting functionality from individual scripts:

- 6-subplot delta plots

- All local vs central comparison

- Comprehensive SK vs FL plots

- Standard FL models comparison plots

- FL vs all local comparison

- Local vs central comparison

"""

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pathlib import Path

import warnings

import argparse

import os

warnings.filterwarnings('ignore')

# ============================================================================

# COMMON HELPER FUNCTIONS

# ============================================================================

def get_comparison_stats(df, model1, model2):

    """Extract comparison statistics between two models"""

    # Look for comparison in both directions

    comparison1 = df[(df['Model_A'] == model1) & (df['Model_B'] == model2)]

    comparison2 = df[(df['Model_A'] == model2) & (df['Model_B'] == model1)]

    

    if not comparison1.empty:

        row = comparison1.iloc[0]

        return {

            'model1_mean': row['Mean_A'],

            'model2_mean': row['Mean_B'],

            'corrected_p': row.get('Corrected_p', None),

            'significant_winner': row['Significant_Winner'] if pd.notna(row['Significant_Winner']) else "None"

        }

    elif not comparison2.empty:

        row = comparison2.iloc[0]

        return {

            'model1_mean': row['Mean_B'],

            'model2_mean': row['Mean_A'],

            'corrected_p': row.get('Corrected_p', None),

            'significant_winner': row['Significant_Winner'] if pd.notna(row['Significant_Winner']) else "None"

        }

    

    return None

def load_all_results(disc_results_dir=None, cup_results_dir=None):

    """Load all CSV result files for both cup and disc tasks

    

    Args:

        disc_results_dir: Path to disc statistical analysis results directory

        cup_results_dir: Path to cup statistical analysis results directory

    """

    datasets = ['binrushed', 'chaksu', 'drishti', 'g1020', 'magrabi', 'messidor', 'origa', 'refuge', 'rimone']

    

    results = {}

    

    for task in ['cup', 'disc']:

        results[task] = {}

        

        # Use provided directories or fall back to defaults

        if task == 'cup':

            base_dir = Path(cup_results_dir) if cup_results_dir else Path('./Statistics/cup')

        else:

            base_dir = Path(disc_results_dir) if disc_results_dir else Path('./Statistics/disc')

        

        if not base_dir.exists():

            print(f"Warning: Directory {base_dir} does not exist")

            continue

        

        for eval_dataset in datasets:

            csv_file = base_dir / f'{eval_dataset}_{task}_pairwise_wilcoxon.csv'

            

            if csv_file.exists():

                try:

                    df = pd.read_csv(csv_file)

                    # Clean column names

                    df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('\r', '')

                    

                    # Fix malformed first column name

                    if len(df.columns) > 0 and 'Model_A' in df.columns[0] and df.columns[0] != 'Model_A':

                        new_columns = list(df.columns)

                        new_columns[0] = 'Model_A'

                        df.columns = new_columns

                    

                    # Clean data values

                    for col in ['Model_A', 'Model_B', 'Better_Model', 'Significant_Winner']:

                        if col in df.columns:

                            df[col] = df[col].astype(str).str.strip()

                    

                    results[task][eval_dataset] = df

                except Exception as e:

                    print(f"Warning: Error reading {csv_file}: {e}")

                    continue

            else:

                print(f"Warning: {csv_file} not found")

    

    return results

# ============================================================================

# PLOT 1: 6-SUBPLOT DELTA PLOTS

# ============================================================================

def convert_significant_winner_6subplot(winner, model1_mean, model2_mean, p_value, model1_name):

    """Convert significant winner to A/B/None format for 6-subplot plots"""

    if winner == "None":

        return 'None'

    

    winner_str = str(winner).lower()

    

    if (model1_name.lower() in winner_str or 

        any(term in winner_str for term in ['central', 'global', 'unweighted', 'fl_finetuned'])):

        if 'fl_finetuned' in winner_str and 'fl_finetuned' not in model1_name:

            return 'B'

        else:

            return 'A'

    else:

        return 'None'

def create_subplot_delta_plot(ax, datasets, model_a_means, model_b_means, significant_winners, 

                             label_a, label_b, title):

    """Create a delta comparison plot in a subplot"""

    

    if not datasets:

        ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes, 

                ha='center', va='center', fontsize=18)

        ax.set_title(title, fontsize=18, fontweight='bold')

        return

    

    deltas = [b - a for a, b in zip(model_a_means, model_b_means)]

    x_positions = range(len(datasets))

    

    for i, (delta, winner) in enumerate(zip(deltas, significant_winners)):

        onsite_wins = False

        other_wins = False

        if winner == 'A' and 'Fine-Tuned Onsite Validation' in label_a:

            onsite_wins = True

        elif winner == 'B' and 'Fine-Tuned Onsite Validation' in label_b:

            onsite_wins = True

        elif winner == 'A' and 'Fine-Tuned Onsite Validation' not in label_a:

            other_wins = True

        elif winner == 'B' and 'Fine-Tuned Onsite Validation' not in label_b:

            other_wins = True

        

        if onsite_wins:

            bar = ax.bar(i, delta, color='lightblue', edgecolor='blue', 

                        hatch='///', linewidth=1, alpha=0.8, width=0.6)

        elif other_wins:

            bar = ax.bar(i, delta, color='lightcoral', edgecolor='red', 

                        hatch='...', linewidth=1, alpha=0.8, width=0.6)

        else:

            bar = ax.bar(i, delta, color='lightgray', edgecolor='gray', 

                        linewidth=1, alpha=0.6, width=0.6)

    

    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_ylabel(f"Δ({label_b} - {label_a})", fontsize=18, labelpad=15)

    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

    ax.set_xticks(x_positions)

    ax.set_xticklabels(datasets, fontsize=17, rotation=45, ha='right')

    ax.tick_params(axis='y', labelsize=17)

    

    return ax

def create_6_subplot_delta_plots(results, output_base_dir):

    """Create separate cup and disc plots with 6 subplots each"""

    

    print("\n=== Creating 6-subplot delta plots ===")

    

    datasets = ['binrushed', 'chaksu', 'drishti', 'g1020', 'magrabi', 'messidor', 'origa', 'refuge', 'rimone']

    

    comparisons = [

        {

            'name': 'Local Models vs Fine-Tuned Onsite Validation',

            'model_a': 'sk',

            'model_b': 'fl_finetuned',

            'label_a': 'Local Model',

            'label_b': 'Fine-Tuned Onsite Validation',

            'subplot_label': '(i)'

        },

        {

            'name': 'Central Model vs Fine-Tuned Onsite Validation',

            'model_a': 'central_baseline', 

            'model_b': 'fl_finetuned',

            'label_a': 'Central Model',

            'label_b': 'Fine-Tuned Onsite Validation',

            'subplot_label': '(ii)'

        },

        {

            'name': 'Global Validation vs Fine-Tuned Onsite Validation',

            'model_a': 'globalVal',

            'model_b': 'fl_finetuned', 

            'label_a': 'Global Validation',

            'label_b': 'Fine-Tuned Onsite Validation',

            'subplot_label': ''

        },

        {

            'name': 'Weighted Global Validation vs Fine-Tuned Onsite Validation',

            'model_a': 'globalVal_weighted_FedAvg',

            'model_b': 'fl_finetuned',

            'label_a': 'Weighted Global Validation',

            'label_b': 'Fine-Tuned Onsite Validation',

            'subplot_label': ''

        },

        {

            'name': 'Onsite Validation vs Fine-Tuned Onsite Validation', 

            'model_a': 'unweightedglobalevalwithlocaltraining',

            'model_b': 'fl_finetuned',

            'label_a': 'Onsite Validation',

            'label_b': 'Fine-Tuned Onsite Validation',

            'subplot_label': ''

        }

    ]

    

    comparison_data = {}

    

    for task in ['cup', 'disc']:

        comparison_data[task] = {}

        

        for comp in comparisons:

            comp_name = comp['name']

            comparison_data[task][comp_name] = {

                'datasets': [],

                'model_a_means': [],

                'model_b_means': [],

                'significant_winners': []

            }

            

            for eval_dataset in datasets:

                if eval_dataset in results[task]:

                    df = results[task][eval_dataset]

                    

                    if comp['model_a'] == 'sk':

                        model1 = f'sk_{eval_dataset}'

                    elif comp['model_a'] == 'fl_finetuned':

                        model1 = f'{eval_dataset}_fl_finetuned'

                    else:

                        model1 = comp['model_a']

                    

                    if comp['model_b'] == 'fl_finetuned':

                        model2 = f'{eval_dataset}_fl_finetuned'

                    elif comp['model_b'] == 'sk':

                        model2 = f'sk_{eval_dataset}'

                    else:

                        model2 = comp['model_b']

                    

                    comparison = get_comparison_stats(df, model1, model2)

                    if comparison:

                        comparison_data[task][comp_name]['datasets'].append(eval_dataset.upper())

                        comparison_data[task][comp_name]['model_a_means'].append(comparison['model1_mean'])

                        comparison_data[task][comp_name]['model_b_means'].append(comparison['model2_mean'])

                        

                        winner = convert_significant_winner_6subplot(

                            comparison['significant_winner'], 

                            comparison['model1_mean'], 

                            comparison['model2_mean'],

                            comparison['corrected_p'],

                            model1

                        )

                        comparison_data[task][comp_name]['significant_winners'].append(winner)

    

    output_dir = output_base_dir / 'onsite_finetuned_comparisons'

    output_dir.mkdir(parents=True, exist_ok=True)

    

    for task in ['cup', 'disc']:

        print(f"Creating 6-subplot comparison for {task.upper()}...")

        

        fig, axes = plt.subplots(2, 3, figsize=(28, 20))

        fig.suptitle(f'Model Performance Comparisons - {task.upper()} Segmentation', 

                     fontsize=24, fontweight='bold', y=0.98)

        

        axes_flat = axes.flatten()

        subplot_positions = [0, 1, 3, 4, 5]

        

        for i, comp in enumerate(comparisons[:5]):

            ax = axes_flat[subplot_positions[i]]

            comp_name = comp['name']

            

            if comp_name in comparison_data[task]:

                data = comparison_data[task][comp_name]

                

                if comp['subplot_label']:

                    subplot_title = f"{comp['subplot_label']} Fine-Tuned Onsite Validation vs {comp['label_a']}"

                else:

                    subplot_title = f"Fine-Tuned Onsite Validation vs {comp['label_a']}"

                create_subplot_delta_plot(

                    ax, data['datasets'], data['model_a_means'], data['model_b_means'],

                    data['significant_winners'], comp['label_a'], comp['label_b'], subplot_title

                )

            else:

                ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes, 

                        ha='center', va='center', fontsize=18)

                if comp['subplot_label']:

                    subplot_title = f"{comp['subplot_label']} Fine-Tuned Onsite Validation vs {comp['label_a']}"

                else:

                    subplot_title = f"Fine-Tuned Onsite Validation vs {comp['label_a']}"

                ax.set_title(subplot_title, fontsize=18, fontweight='bold')

        

        axes_flat[2].set_visible(False)

        fig.text(0.5, 0.05, '(iii)', ha='center', va='bottom', fontsize=22, fontweight='bold')

        

        from matplotlib.patches import Patch

        legend_elements = [

            Patch(facecolor='lightblue', edgecolor='blue', hatch='///', 

                  label='Fine-Tuned Onsite Validation significantly better', alpha=0.8),

            Patch(facecolor='lightcoral', edgecolor='red', hatch='...', 

                  label='Other model significantly better', alpha=0.8),

            Patch(facecolor='lightgray', edgecolor='gray', 

                  label='No significant difference', alpha=0.6)

        ]

        

        plt.tight_layout()

        plt.subplots_adjust(top=0.88, bottom=0.15, hspace=0.5, wspace=0.35)

        

        fig.legend(handles=legend_elements, loc='upper right', ncol=1, 

                   bbox_to_anchor=(0.98, 0.95), fontsize=17)

        

        output_path = output_dir / f'onsite_finetuned_vs_all_models_{task}.png'

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

        print(f"  Saved: {output_path}")

        plt.close()

# ============================================================================

# PLOT 2: ALL LOCAL VS CENTRAL COMPARISON

# ============================================================================

def convert_significant_winner_local_central(winner, local_model_name, central_model_name):

    """Convert significant winner to A/B format for local vs central"""

    if winner == "None":

        return 'None'

    

    winner_str = str(winner).lower()

    

    if 'sk_' in winner_str:

        return 'A'

    elif 'central_baseline' in winner_str:

        return 'B'

    else:

        return 'None'

def create_subplot_local_vs_central(ax, training_datasets, local_means, central_means, 

                                   significant_winners, eval_dataset):

    """Create a single subplot for local vs central comparison"""

    

    if not training_datasets or len(training_datasets) == 0:

        ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes, 

                ha='center', va='center', fontsize=10)

        ax.set_title(f'Eval: {eval_dataset}', fontsize=11, fontweight='bold')

        return

    

    deltas = [b - a for a, b in zip(local_means, central_means)]

    x_positions = range(len(training_datasets))

    

    colors = []

    for winner in significant_winners:

        if winner == 'A':

            colors.append(('lightcoral', 'red', '...'))

        elif winner == 'B':

            colors.append(('lightblue', 'blue', '///'))

        else:

            colors.append(('lightgray', 'gray', ''))

    

    bars = []

    for i, (delta, (facecolor, edgecolor, hatch)) in enumerate(zip(deltas, colors)):

        bar = ax.bar(i, delta, color=facecolor, edgecolor=edgecolor, 

                    hatch=hatch, alpha=0.8, linewidth=1.5 if hatch else 1)

        bars.append(bar)

    

    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_ylabel("Δ(Central - Local)", fontsize=9, rotation=90)

    ax.set_title(f'Eval: {eval_dataset}', fontsize=11, fontweight='bold')

    ax.set_xticks(x_positions)

    ax.set_xticklabels([d.upper() for d in training_datasets], 

                       fontsize=8, rotation=45, ha='right')

    ax.tick_params(axis='y', labelsize=8)

def create_all_local_vs_central_comparisons(results, output_base_dir):

    """Create all local vs central baseline comparison plots"""

    

    print("\n=== Creating all local vs central comparison plots ===")

    

    datasets = ['binrushed', 'chaksu', 'drishti', 'g1020', 'magrabi', 'messidor', 'origa', 'refuge', 'rimone']

    

    output_dir = output_base_dir / 'central_vs_local_by_dataset'

    output_dir.mkdir(exist_ok=True)

    

    for task in ['cup', 'disc']:

        comparison_data = {}

        

        for eval_dataset in datasets:

            comparison_data[eval_dataset] = {

                'training_datasets': [],

                'local_means': [],

                'central_means': [],

                'significant_winners': []

            }

            

            if eval_dataset in results[task]:

                df = results[task][eval_dataset]

                

                for train_dataset in datasets:

                    local_model = f'sk_{train_dataset}'

                    central_model = 'central_baseline'

                    

                    comparison = get_comparison_stats(df, local_model, central_model)

                    if comparison:

                        comparison_data[eval_dataset]['training_datasets'].append(train_dataset)

                        comparison_data[eval_dataset]['local_means'].append(comparison['model1_mean'])

                        comparison_data[eval_dataset]['central_means'].append(comparison['model2_mean'])

                        

                        winner = convert_significant_winner_local_central(

                            comparison['significant_winner'], 

                            local_model, 

                            central_model

                        )

                        comparison_data[eval_dataset]['significant_winners'].append(winner)

        

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))

        fig.suptitle(f'Central Model vs Local Models - {task.upper()} Segmentation', 

                     fontsize=16, fontweight='bold', y=0.95)

        

        for i, eval_dataset in enumerate(datasets):

            row = i // 3

            col = i % 3

            ax = axes[row, col]

            

            data = comparison_data[eval_dataset]

            

            if data['training_datasets']:

                create_subplot_local_vs_central(

                    ax, data['training_datasets'], data['local_means'], data['central_means'],

                    data['significant_winners'], eval_dataset.upper()

                )

            else:

                ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes, 

                        ha='center', va='center', fontsize=10)

                ax.set_title(f'Eval: {eval_dataset.upper()}', fontsize=11, fontweight='bold')

        

        from matplotlib.patches import Patch

        legend_elements = [

            Patch(facecolor='lightblue', edgecolor='blue', hatch='///', 

                  label='Central Model significantly better', alpha=0.8),

            Patch(facecolor='lightcoral', edgecolor='red', hatch='...', 

                  label='Local model significantly better', alpha=0.8),

            Patch(facecolor='lightgray', edgecolor='gray', 

                  label='No significant difference', alpha=0.6)

        ]

        

        fig.legend(handles=legend_elements, loc='lower center', ncol=3, 

                   bbox_to_anchor=(0.5, 0.02), fontsize=10)

        

        plt.tight_layout()

        plt.subplots_adjust(top=0.90, bottom=0.12)

        

        output_path = output_dir / f'central_vs_local_models_by_dataset_{task}.png'

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

        print(f"  Saved: {output_path}")

        plt.close()

# ============================================================================

# PLOT 3: COMPREHENSIVE SK VS FL PLOTS

# ============================================================================

def convert_significant_winner_sk_fl(winner, fl_mean, sk_mean, p_value):

    """Convert significant winner to Onsite/Local/None format"""

    if winner == "None":

        return 'None'

    

    winner_str = str(winner).lower()

    

    if 'fl_finetuned' in winner_str:

        return 'Onsite'

    elif 'sk_' in winner_str:

        return 'Local'

    else:

        return 'None'

def create_comprehensive_sk_vs_fl_plots(results, output_base_dir):

    """Create comprehensive SK vs FL fine-tuned comparison plots"""

    

    print("\n=== Creating comprehensive local model vs fine-tuned onsite validation plots ===")

    

    datasets = ['binrushed', 'chaksu', 'drishti', 'g1020', 'magrabi', 'messidor', 'origa', 'refuge', 'rimone']

    dataset_labels = [d.upper() for d in datasets]

    

    output_dir = output_base_dir / 'local_vs_onsite_finetuned'

    output_dir.mkdir(parents=True, exist_ok=True)

    

    for task in ['cup', 'disc']:

        print(f"Creating comprehensive plot for {task.upper()} segmentation...")

        

        fig, axes = plt.subplots(3, 3, figsize=(36, 30))

        fig.suptitle(f'Local Model vs Fine-Tuned Onsite Validation Models - {task.upper()} Segmentation', 

                 fontsize=36, fontweight='bold', y=0.98)

        

        axes_flat = axes.flatten()

        

        for eval_idx, eval_dataset in enumerate(datasets):

            ax = axes_flat[eval_idx]

            

            if eval_dataset in results[task]:

                df = results[task][eval_dataset]

                

                fl_means = []

                sk_means = []

                deltas = []

                colors = []

                

                for fl_dataset in datasets:

                    fl_model = f'{fl_dataset}_fl_finetuned'

                    sk_model = f'sk_{fl_dataset}'

                    

                    comparison = get_comparison_stats(df, fl_model, sk_model)

                    

                    if comparison:

                        fl_means.append(comparison['model1_mean'])

                        sk_means.append(comparison['model2_mean'])

                        

                        delta = comparison['model1_mean'] - comparison['model2_mean']

                        deltas.append(delta)

                        

                        winner = convert_significant_winner_sk_fl(

                            comparison['significant_winner'],

                            comparison['model1_mean'],

                            comparison['model2_mean'],

                            comparison['corrected_p']

                        )

                        

                        if winner == 'Onsite':

                            colors.append(('lightblue', 'blue', '///'))

                        elif winner == 'Local':

                            colors.append(('lightcoral', 'red', '...'))

                        else:

                            colors.append(('lightgray', 'gray', ''))

                    else:

                        fl_means.append(0)

                        sk_means.append(0)

                        deltas.append(0)

                        colors.append(('white', 'black', ''))

                

                x_positions = range(len(datasets))

                bars = []

                for i, (delta, (facecolor, edgecolor, hatch)) in enumerate(zip(deltas, colors)):

                    bar = ax.bar(i, delta, color=facecolor, edgecolor=edgecolor, 

                                hatch=hatch, alpha=0.8, linewidth=1.5 if hatch else 1)

                    bars.extend(bar)

                

                ax.axhline(0, color="black", linewidth=0.8)

                ax.set_title(f'Evaluated on {eval_dataset.upper()}', fontsize=30, fontweight='bold', pad=20)

                ax.set_xlabel('', fontsize=0)

                ax.set_ylabel('Δ(Fine-Tuned Onsite Validation - Local Model)', fontsize=18, labelpad=25)

                ax.set_xticks(x_positions)

                ax.set_xticklabels(dataset_labels, rotation=45, ha='right', fontsize=17)

                ax.tick_params(axis='y', labelsize=17)

                

                for i, (bar, delta) in enumerate(zip(bars, deltas)):

                    if delta != 0:

                        height = bar.get_height()

                        ax.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.003),

                               f'{delta:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 

                               fontsize=14, rotation=0)

            else:

                ax.text(0.5, 0.5, f'No Data Available\nfor {eval_dataset.upper()}', 

                       transform=ax.transAxes, ha='center', va='center', fontsize=18)

                ax.set_title(f'Evaluated on {eval_dataset.upper()}', fontsize=30, fontweight='bold', pad=20)

        

        from matplotlib.patches import Patch

        legend_elements = [

            Patch(facecolor='lightblue', edgecolor='blue', hatch='///',

                  label='Fine-Tuned Onsite Validation significantly better', alpha=0.8),

            Patch(facecolor='lightcoral', edgecolor='red', hatch='...',

                  label='Local Model significantly better', alpha=0.8),

            Patch(facecolor='lightgray', edgecolor='gray', 

                  label='No significant difference', alpha=0.8)

        ]

        

        plt.tight_layout()

        plt.subplots_adjust(top=0.90, bottom=0.15, hspace=0.3, wspace=0.2)

        

        fig.legend(handles=legend_elements, loc='lower center', ncol=3, 

                   bbox_to_anchor=(0.5, 0.08), fontsize=21)

        

        output_path = output_dir / f'local_vs_onsite_finetuned_comprehensive_{task}.png'

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

        print(f"  Saved: {output_path}")

        plt.close()

# ============================================================================

# PLOT 4: STANDARD FL MODELS COMPARISON PLOTS

# ============================================================================

def convert_significant_winner_fl_baseline(winner, model1_name, model2_name):

    """Convert significant winner to A/B format for Standard FL models comparison"""

    if winner == "None":

        return 'None'

    

    winner_str = str(winner).lower()

    

    if ('sk_' in winner_str and 'sk' in model2_name.lower()):

        return 'B'

    elif ('central_baseline' in winner_str and 'central_baseline' in model2_name.lower()):

        return 'B'

    elif (model1_name.lower() in winner_str or 

          any(term in winner_str for term in ['global', 'unweighted'])):

        return 'A'

    else:

        return 'None'

def create_subplot_fl_comparison(ax, datasets, model_a_means, model_b_means, 

                                significant_winners, label_a, label_b, title):

    """Create a single subplot for Standard FL models comparison"""

    

    if not datasets or len(datasets) == 0:

        ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes, 

                ha='center', va='center', fontsize=12)

        ax.set_title(title, fontsize=11, fontweight='bold')

        return

    

    deltas = [a - b for a, b in zip(model_a_means, model_b_means)]

    x_positions = range(len(datasets))

    

    for i, (delta, winner) in enumerate(zip(deltas, significant_winners)):

        if winner == 'A':

            ax.bar(i, delta, color='lightblue', edgecolor='blue', 

                  hatch='///', linewidth=1.5, alpha=0.8, width=0.6)

        elif winner == 'B':

            ax.bar(i, delta, color='lightcoral', edgecolor='red',

                  hatch='...', linewidth=1.5, alpha=0.8, width=0.6)

        else:

            ax.bar(i, delta, color='lightgray', edgecolor='gray',

                  linewidth=1, alpha=0.6, width=0.6)

    

    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_ylabel(f"Δ({label_a} - {label_b})", fontsize=12)

    ax.set_title(title, fontsize=11, fontweight='bold')

    ax.set_xticks(x_positions)

    ax.set_xticklabels(datasets, fontsize=12, rotation=45, ha='right')

    ax.tick_params(axis='y', labelsize=11)

def create_fl_baseline_comparison_plots(results, output_base_dir):

    """Create Standard FL models comparison plots"""

    

    print("\n=== Creating Standard FL models comparison plots ===")

    

    datasets = ['binrushed', 'chaksu', 'drishti', 'g1020', 'magrabi', 'messidor', 'origa', 'refuge', 'rimone']

    

    fl_models = [

        {'name': 'globalVal', 'label': 'Global Validation'},

        {'name': 'globalVal_weighted_FedAvg', 'label': 'Weighted Global Validation'},

        {'name': 'unweightedglobalevalwithlocaltraining', 'label': 'Onsite Validation'}

    ]

    

    baselines = [

        {'name': 'sk', 'label': 'Local Model'},

        {'name': 'central_baseline', 'label': 'Central Model'}

    ]

    

    output_dir = output_base_dir / 'fl_base_models_comparison'

    output_dir.mkdir(exist_ok=True)

    

    for task in ['cup', 'disc']:

        comparison_data = {}

        

        for baseline in baselines:

            comparison_data[baseline['name']] = {}

            

            for fl_model in fl_models:

                comparison_data[baseline['name']][fl_model['name']] = {

                    'datasets': [],

                    'model_a_means': [],

                    'model_b_means': [],

                    'significant_winners': []

                }

                

                for eval_dataset in datasets:

                    if eval_dataset in results[task]:

                        df = results[task][eval_dataset]

                        

                        if baseline['name'] == 'sk':

                            baseline_model = f'sk_{eval_dataset}'

                        else:

                            baseline_model = baseline['name']

                        

                        fl_model_name = fl_model['name']

                        

                        comparison = get_comparison_stats(df, fl_model_name, baseline_model)

                        if comparison:

                            comparison_data[baseline['name']][fl_model['name']]['datasets'].append(eval_dataset.upper())

                            comparison_data[baseline['name']][fl_model['name']]['model_a_means'].append(comparison['model1_mean'])

                            comparison_data[baseline['name']][fl_model['name']]['model_b_means'].append(comparison['model2_mean'])

                            

                            winner = convert_significant_winner_fl_baseline(

                                comparison['significant_winner'], 

                                fl_model_name, 

                                baseline_model

                            )

                            comparison_data[baseline['name']][fl_model['name']]['significant_winners'].append(winner)

        

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        fig.suptitle(f'Standard FL Models vs Central and Local Models - {task.upper()} Segmentation', 

                     fontsize=16, fontweight='bold', y=0.95)

        

        baseline = baselines[0]

        for i, fl_model in enumerate(fl_models):

            ax = axes[0, i]

            data = comparison_data[baseline['name']][fl_model['name']]

            

            if data['datasets']:

                title = f"{fl_model['label']} vs {baseline['label']}"

                create_subplot_fl_comparison(

                    ax, data['datasets'], data['model_a_means'], data['model_b_means'],

                    data['significant_winners'], fl_model['label'], baseline['label'], title

                )

            else:

                ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes, 

                        ha='center', va='center', fontsize=12)

                ax.set_title(f"{fl_model['label']} vs {baseline['label']}", fontsize=11, fontweight='bold')

        

        baseline = baselines[1]

        for i, fl_model in enumerate(fl_models):

            ax = axes[1, i]

            data = comparison_data[baseline['name']][fl_model['name']]

            

            if data['datasets']:

                title = f"{fl_model['label']} vs {baseline['label']}"

                create_subplot_fl_comparison(

                    ax, data['datasets'], data['model_a_means'], data['model_b_means'],

                    data['significant_winners'], fl_model['label'], baseline['label'], title

                )

            else:

                ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes, 

                        ha='center', va='center', fontsize=12)

                ax.set_title(f"{fl_model['label']} vs {baseline['label']}", fontsize=11, fontweight='bold')

        

        from matplotlib.patches import Patch

        legend_elements = [

            Patch(facecolor='lightblue', edgecolor='blue', hatch='///', 

                  label='standard FL model significantly better', alpha=0.8),

            Patch(facecolor='lightcoral', edgecolor='red', hatch='...', 

                  label='Other model significantly better', alpha=0.8),

            Patch(facecolor='lightgray', edgecolor='gray', 

                  label='No significant difference', alpha=0.6)

        ]

        

        fig.legend(handles=legend_elements, loc='lower center', ncol=3, 

                   bbox_to_anchor=(0.5, 0.02), fontsize=10)

        

        plt.tight_layout()

        plt.subplots_adjust(top=0.88, bottom=0.15)

        

        output_path = output_dir / f'fl_base_models_vs_local_and_central_{task}.png'

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

        print(f"  Saved: {output_path}")

        plt.close()

# ============================================================================

# PLOT 5: FL VS ALL LOCAL COMPARISON

# ============================================================================

def convert_significant_winner_fl_local(winner, fl_model_name, local_model_name):

    """Convert significant winner to A/B format for FL vs local"""

    if winner == "None":

        return 'None'

    

    winner_str = str(winner).lower()

    

    if (fl_model_name.lower() in winner_str or 

        any(term in winner_str for term in ['global', 'unweighted'])):

        return 'A'

    elif 'sk_' in winner_str:

        return 'B'

    else:

        return 'None'

def create_subplot_fl_vs_local(ax, training_datasets, fl_means, local_means, 

                              significant_winners, eval_dataset, fl_label):

    """Create a single subplot for FL vs local comparison"""

    

    if not training_datasets or len(training_datasets) == 0:

        ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes, 

                ha='center', va='center', fontsize=10)

        ax.set_title(f'Eval: {eval_dataset}', fontsize=11, fontweight='bold')

        return

    

    deltas = [a - b for a, b in zip(fl_means, local_means)]

    x_positions = range(len(training_datasets))

    

    colors = []

    for winner in significant_winners:

        if winner == 'A':

            colors.append(('lightblue', 'blue', '///'))

        elif winner == 'B':

            colors.append(('lightcoral', 'red', '...'))

        else:

            colors.append(('lightgray', 'gray', ''))

    

    bars = []

    for i, (delta, (facecolor, edgecolor, hatch)) in enumerate(zip(deltas, colors)):

        bar = ax.bar(i, delta, color=facecolor, edgecolor=edgecolor, 

                    hatch=hatch, alpha=0.8, linewidth=1.5 if hatch else 1)

        bars.append(bar)

    

    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_ylabel(f"Δ({fl_label} - Local)", fontsize=12, rotation=90)

    ax.set_title(f'Eval: {eval_dataset}', fontsize=11, fontweight='bold')

    ax.set_xticks(x_positions)

    ax.set_xticklabels([d.upper() for d in training_datasets], 

                       fontsize=12, rotation=45, ha='right')

    ax.tick_params(axis='y', labelsize=11)

def create_fl_vs_all_local_comparisons(results, output_base_dir):

    """Create FL vs all local comparison plots"""

    

    print("\n=== Creating FL vs all local comparison plots ===")

    

    datasets = ['binrushed', 'chaksu', 'drishti', 'g1020', 'magrabi', 'messidor', 'origa', 'refuge', 'rimone']

    

    fl_models = [

        {'name': 'globalVal', 'label': 'Global Validation'},

        {'name': 'globalVal_weighted_FedAvg', 'label': 'Weighted Global Validation'},

        {'name': 'unweightedglobalevalwithlocaltraining', 'label': 'Onsite Validation'}

    ]

    

    output_dir = output_base_dir / 'fl_models_vs_local'

    output_dir.mkdir(exist_ok=True)

    

    for fl_model in fl_models:

        for task in ['cup', 'disc']:

            comparison_data = {}

            

            for eval_dataset in datasets:

                comparison_data[eval_dataset] = {

                    'training_datasets': [],

                    'fl_means': [],

                    'local_means': [],

                    'significant_winners': []

                }

                

                if eval_dataset in results[task]:

                    df = results[task][eval_dataset]

                    

                    for train_dataset in datasets:

                        local_model = f'sk_{train_dataset}'

                        fl_model_name = fl_model['name']

                        

                        comparison = get_comparison_stats(df, fl_model_name, local_model)

                        if comparison:

                            comparison_data[eval_dataset]['training_datasets'].append(train_dataset)

                            comparison_data[eval_dataset]['fl_means'].append(comparison['model1_mean'])

                            comparison_data[eval_dataset]['local_means'].append(comparison['model2_mean'])

                            

                            winner = convert_significant_winner_fl_local(

                                comparison['significant_winner'], 

                                fl_model_name, 

                                local_model

                            )

                            comparison_data[eval_dataset]['significant_winners'].append(winner)

            

            fig, axes = plt.subplots(3, 3, figsize=(15, 12))

            fig.suptitle(f'{fl_model["label"]} vs Local Models - {task.upper()} Segmentation', 

                         fontsize=16, fontweight='bold', y=0.95)

            

            for i, eval_dataset in enumerate(datasets):

                row = i // 3

                col = i % 3

                ax = axes[row, col]

                

                data = comparison_data[eval_dataset]

                

                if data['training_datasets']:

                    create_subplot_fl_vs_local(

                        ax, data['training_datasets'], data['fl_means'], data['local_means'],

                        data['significant_winners'], eval_dataset.upper(), fl_model['label']

                    )

                else:

                    ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes, 

                            ha='center', va='center', fontsize=10)

                    ax.set_title(f'Eval: {eval_dataset.upper()}', fontsize=11, fontweight='bold')

            

            from matplotlib.patches import Patch

            legend_elements = [

                Patch(facecolor='lightblue', edgecolor='blue', hatch='///', 

                      label=f'{fl_model["label"]} significantly better', alpha=0.8),

                Patch(facecolor='lightcoral', edgecolor='red', hatch='...', 

                      label='Local model significantly better', alpha=0.8),

                Patch(facecolor='lightgray', edgecolor='gray', 

                      label='No significant difference', alpha=0.6)

            ]

            

            fig.legend(handles=legend_elements, loc='lower center', ncol=3, 

                       bbox_to_anchor=(0.5, 0.02), fontsize=10)

            

            plt.tight_layout()

            plt.subplots_adjust(top=0.90, bottom=0.12)

            

            fl_name_clean = fl_model['name'].replace('globalVal_weighted_FedAvg', 'weighted_global_validation').replace('unweightedglobalevalwithlocaltraining', 'onsite_validation').replace('globalVal', 'global_validation')

            output_path = output_dir / f'{fl_name_clean}_vs_local_models_{task}.png'

            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

            print(f"  Saved: {output_path}")

            plt.close()

# ============================================================================

# PLOT 6: LOCAL VS CENTRAL COMPARISON

# ============================================================================

def create_subplot_simple_local_vs_central(ax, datasets, model_a_means, model_b_means, 

                                   significant_winners, title):

    """Create a single subplot for local vs central comparison"""

    

    if not datasets or len(datasets) == 0:

        ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes, 

                ha='center', va='center', fontsize=12)

        ax.set_title(title, fontsize=14, fontweight='bold')

        return

    

    deltas = [b - a for a, b in zip(model_a_means, model_b_means)]

    x_positions = range(len(datasets))

    

    for i, (delta, winner) in enumerate(zip(deltas, significant_winners)):

        if winner == 'A':

            ax.bar(i, delta, color='lightcoral', edgecolor='red', 

                  hatch='...', linewidth=1.5, alpha=0.8, width=0.6)

        elif winner == 'B':

            ax.bar(i, delta, color='lightblue', edgecolor='blue',

                  hatch='///', linewidth=1.5, alpha=0.8, width=0.6)

        else:

            ax.bar(i, delta, color='lightgray', edgecolor='gray',

                  linewidth=1, alpha=0.6, width=0.6)

    

    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_ylabel("Δ(Central Model - Local Models)", fontsize=12)

    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_xticks(x_positions)

    ax.set_xticklabels(datasets, fontsize=10, rotation=45, ha='right')

    ax.tick_params(axis='y', labelsize=10)

def create_local_vs_central_comparison(results, output_base_dir):

    """Create local vs central baseline comparison plot"""

    

    print("\n=== Creating local vs central comparison plot ===")

    

    datasets = ['binrushed', 'chaksu', 'drishti', 'g1020', 'magrabi', 'messidor', 'origa', 'refuge', 'rimone']

    

    output_dir = output_base_dir / 'local_vs_central'

    output_dir.mkdir(exist_ok=True)

    

    comparison_data = {}

    

    for task in ['cup', 'disc']:

        comparison_data[task] = {

            'datasets': [],

            'local_means': [],

            'central_means': [],

            'significant_winners': []

        }

        

        for eval_dataset in datasets:

            if eval_dataset in results[task]:

                df = results[task][eval_dataset]

                

                local_model = f'sk_{eval_dataset}'

                central_model = 'central_baseline'

                

                comparison = get_comparison_stats(df, local_model, central_model)

                if comparison:

                    comparison_data[task]['datasets'].append(eval_dataset.upper())

                    comparison_data[task]['local_means'].append(comparison['model1_mean'])

                    comparison_data[task]['central_means'].append(comparison['model2_mean'])

                    

                    winner = convert_significant_winner_local_central(

                        comparison['significant_winner'], 

                        local_model, 

                        central_model

                    )

                    comparison_data[task]['significant_winners'].append(winner)

    

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    fig.suptitle('Central Model vs Local Models Comparison', 

                 fontsize=16, fontweight='bold', y=0.95)

    

    for i, task in enumerate(['cup', 'disc']):

        ax = axes[i]

        data = comparison_data[task]

        

        if data['datasets']:

            title = f'{task.upper()} Segmentation'

            create_subplot_simple_local_vs_central(

                ax, data['datasets'], data['local_means'], data['central_means'],

                data['significant_winners'], title

            )

        else:

            ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes, 

                    ha='center', va='center', fontsize=12)

            ax.set_title(f'{task.upper()} Segmentation', fontsize=14, fontweight='bold')

    

    from matplotlib.patches import Patch

    legend_elements = [

        Patch(facecolor='lightblue', edgecolor='blue', hatch='///', 

              label='Central Model significantly better', alpha=0.8),

        Patch(facecolor='lightcoral', edgecolor='red', hatch='...', 

              label='Local Models significantly better', alpha=0.8),

        Patch(facecolor='lightgray', edgecolor='gray', 

              label='No significant difference', alpha=0.6)

    ]

    

    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 

               bbox_to_anchor=(0.5, 0.02), fontsize=11)

    

    plt.tight_layout()

    plt.subplots_adjust(top=0.88, bottom=0.25)

    

    output_path = output_dir / 'local_vs_central_model.png'

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"  Saved: {output_path}")

    plt.close()

# ============================================================================

# PLOT 7: OUT-OF-DISTRIBUTION WINS HEATMAP

# ============================================================================

def extract_out_of_distribution_wins(disc_results_dir=None, cup_results_dir=None):

    """Extract out-of-distribution wins for all model types vs SK models

    

    Args:

        disc_results_dir: Path to disc statistical analysis results directory

        cup_results_dir: Path to cup statistical analysis results directory

    """

    

    datasets = ['binrushed', 'chaksu', 'drishti', 'g1020', 'magrabi', 'messidor', 'origa', 'refuge', 'rimone']

    sk_models = [f'sk_{dataset}' for dataset in datasets]

    

    # Model types to analyze

    model_types = {

        'Central Model': 'central_baseline',

        'Onsite Validation': 'unweightedglobalevalwithlocaltraining', 

        'Global Validation': 'globalVal',

        'Weighted Global Validation': 'globalVal_weighted_FedAvg'

    }

    

    # Add FL finetuned models (these are site-specific)

    for dataset in datasets:

        model_types[f'Fine-Tuned Onsite Validation ({dataset.upper()})'] = f'{dataset}_fl_finetuned'

    

    results = {}

    

    for task in ['cup', 'disc']:

        results[task] = {}

        

        # Use provided directories or fall back to defaults

        if task == 'cup':

            base_dir = Path(cup_results_dir) if cup_results_dir else Path('./Statistics/cup')

        else:

            base_dir = Path(disc_results_dir) if disc_results_dir else Path('./Statistics/disc')

        

        if not base_dir.exists():

            print(f"Warning: Directory {base_dir} does not exist for task {task}")

            continue

        

        # Initialize results matrix: rows=model_types, columns=SK models

        win_matrix = np.zeros((len(model_types), len(sk_models)), dtype=int)

        model_names = list(model_types.keys())

        

        # For each evaluation dataset

        for eval_dataset in datasets:

            csv_file = base_dir / f'{eval_dataset}_{task}_pairwise_wilcoxon.csv'

            

            if csv_file.exists():

                try:

                    df = pd.read_csv(csv_file)

                    

                    # Clean column names

                    df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('\r', '')

                    

                    # Fix malformed first column name

                    if len(df.columns) > 0 and 'Model_A' in df.columns[0] and df.columns[0] != 'Model_A':

                        new_columns = list(df.columns)

                        new_columns[0] = 'Model_A'

                        df.columns = new_columns

                    

                    # Clean data values

                    for col in ['Model_A', 'Model_B', 'Better_Model', 'Significant_Winner']:

                        if col in df.columns:

                            df[col] = df[col].astype(str).str.strip()

                    

                    # For each model type

                    for model_idx, (model_name, model_id) in enumerate(model_types.items()):

                        

                        # For each SK model

                        for sk_idx, sk_model in enumerate(sk_models):

                            sk_dataset = sk_model.replace('sk_', '')

                            

                            # Only count out-of-distribution comparisons

                            # (skip if evaluation dataset matches SK model dataset)

                            if eval_dataset != sk_dataset:

                                comparison = get_comparison_winner_heatmap(df, model_id, sk_model)

                                if comparison == model_id:

                                    win_matrix[model_idx, sk_idx] += 1

                

                except Exception as e:

                    print(f"Warning: Error reading {csv_file}: {e}")

                    continue

        

        results[task] = {

            'win_matrix': win_matrix,

            'model_names': model_names,

            'sk_models': sk_models

        }

    

    return results

def get_comparison_winner_heatmap(df, model1, model2):

    """Get the winner from comparison between two models for heatmap"""

    comparison1 = df[(df['Model_A'] == model1) & (df['Model_B'] == model2)]

    comparison2 = df[(df['Model_A'] == model2) & (df['Model_B'] == model1)]

    

    if not comparison1.empty:

        row = comparison1.iloc[0]

        winner = row['Significant_Winner'] if pd.notna(row['Significant_Winner']) and str(row['Significant_Winner']) != 'nan' and str(row['Significant_Winner']).strip() != 'None' else None

        return winner

    elif not comparison2.empty:

        row = comparison2.iloc[0]

        winner = row['Significant_Winner'] if pd.notna(row['Significant_Winner']) and str(row['Significant_Winner']) != 'nan' and str(row['Significant_Winner']).strip() != 'None' else None

        return winner

    

    return None

def create_ood_wins_heatmaps(output_base_dir, disc_results_dir=None, cup_results_dir=None):

    """Create out-of-distribution wins heatmaps for CUP and DISC tasks"""

    

    print("\n=== Creating cross-site performance heatmaps ===")

    

    print("Extracting win data...")

    results = extract_out_of_distribution_wins(disc_results_dir, cup_results_dir)

    

    output_dir = output_base_dir / 'cross_site_performance'

    output_dir.mkdir(exist_ok=True)

    

    for task in ['cup', 'disc']:

        print(f"Creating heatmap for {task.upper()}...")

        

        fig, ax = plt.subplots(figsize=(16, 12))

        

        win_matrix = results[task]['win_matrix']

        model_names = results[task]['model_names']

        sk_models = results[task]['sk_models']

        

        # Remove 'sk_' prefix from SK model names for display

        sk_model_labels = [sk_model.replace('sk_', '').upper() for sk_model in sk_models]

        

        # Create heatmap

        sns.heatmap(win_matrix,

                    xticklabels=sk_model_labels,

                    yticklabels=model_names,

                    annot=True,

                    fmt='d',

                    cmap='YlOrRd',

                    cbar=False,

                    ax=ax,

                    square=False)

        

        ax.set_title(f'{task.upper()} Segmentation Task\nOut-of-Distribution Wins vs Local Models', 

                    fontsize=18, fontweight='bold', pad=30)

        ax.set_xlabel('Local Models', fontsize=14, fontweight='bold')

        ax.set_ylabel('Model Types', fontsize=14, fontweight='bold')

        

        # Rotate x-axis labels for better readability

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)

        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

        

        # Add total wins per model type on the right

        row_totals = win_matrix.sum(axis=1)

        for i, total in enumerate(row_totals):

            ax.text(win_matrix.shape[1] + 0.5, i + 0.5, f'Total: {int(total)}', 

                   ha='left', va='center', fontweight='bold', fontsize=9)

        

        # Add explanation text

        explanation = ("Number of Wins: Number of times each model type significantly outperformed\n"

                      "local models when evaluated on datasets different from the local model's training dataset.\n"

                      "Each Fine-Tuned Onsite Validation model is compared against all local models for cross-site evaluation.\n"

                      "Higher values indicate better cross-site generalization ability.")

        

        fig.text(0.5, 0.02, explanation, ha='center', va='bottom', fontsize=11, 

                 style='italic', wrap=True, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

        

        plt.tight_layout(rect=[0, 0.15, 1, 0.92])

        

        output_path = output_dir / f'cross_site_wins_{task}.png'

        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

        plt.close(fig)

        

        print(f"  Saved: {output_path}")

    

    # Create summary table

    print("Creating summary performance table...")

    create_ood_summary_table(results, output_dir)

def create_ood_summary_table(results, output_dir):

    """Create summary table with total wins per model type for both tasks"""

    

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.axis('tight')

    ax.axis('off')

    

    # Calculate totals

    summary_data = []

    model_names = results['cup']['model_names']

    

    for i, model_name in enumerate(model_names):

        cup_total = int(results['cup']['win_matrix'][i].sum())

        disc_total = int(results['disc']['win_matrix'][i].sum())

        overall_total = cup_total + disc_total

        

        summary_data.append([model_name, cup_total, disc_total, overall_total])

    

    # Sort by overall total (descending)

    summary_data.sort(key=lambda x: x[3], reverse=True)

    

    # Create table with custom column widths (first column wider)

    col_headers = ['Model Type', 'CUP Wins', 'DISC Wins', 'Total Wins']

    col_widths = [0.5, 0.15, 0.15, 0.2]  # First column is wider

    

    table = ax.table(cellText=summary_data,

                    colLabels=col_headers,

                    cellLoc='center',

                    loc='center',

                    bbox=[0, 0, 1, 1],

                    colWidths=col_widths)

    

    # Style the table

    table.auto_set_font_size(False)

    table.set_fontsize(12)

    table.scale(1, 2)

    

    # Color header

    for j in range(len(col_headers)):

        header_cell = table[0, j]

        header_cell.set_facecolor('#4472C4')

        header_cell.set_text_props(weight='bold', color='white')

        header_cell.set_height(0.15)

    

    # Color cells based on performance

    max_total = max([row[3] for row in summary_data]) if summary_data else 1

    for i, row in enumerate(summary_data):

        for j in range(len(col_headers)):

            cell = table[i+1, j]

            cell.set_edgecolor('black')

            cell.set_linewidth(1)

            cell.set_height(0.12)

            

            if j == 0:  # Model name column

                cell.set_text_props(weight='bold', ha='left')

                cell.set_facecolor('#f8f9fa')

            elif j == 3:  # Total column - color by performance

                intensity = row[3] / max_total if max_total > 0 else 0

                cell.set_facecolor(plt.cm.YlOrRd(intensity))

                cell.set_text_props(weight='bold')

    

    plt.suptitle('Performance Summary\nTotal Wins Against Local Models', 

                fontsize=16, fontweight='bold', y=0.95)

    

    # Add note

    note_text = ("Models ranked by total number of wins. Each Fine-Tuned Onsite Validation model\n"

                "is compared against all local models for comprehensive cross-site evaluation.\n"

                "Higher values indicate better generalization across different datasets.")

    plt.figtext(0.5, 0.05, note_text, ha='center', fontsize=10, style='italic', color='gray')

    

    plt.tight_layout(rect=[0, 0.12, 1, 0.9])

    

    summary_path = output_dir / 'cross_site_performance_summary.png'

    fig.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')

    plt.close(fig)

    

    print(f"  Saved: {summary_path}")

    

    # Print statistics

    print("\n  Performance Analysis:")

    for row in summary_data:

        print(f"    {row[0]:<40} | CUP: {row[1]:2d} | DISC: {row[2]:2d} | Total: {row[3]:2d}")

# ============================================================================

# MAIN FUNCTION

# ============================================================================

def parse_args():

    """Parse command line arguments"""

    parser = argparse.ArgumentParser(

        description="Generate comprehensive statistical plots from Friedman/Wilcoxon test results"

    )

    parser.add_argument(

        "--disc_results_dir",

        type=str,

        default=None,

        help="Directory containing disc segmentation statistical analysis results (default: Statistics/disc)"

    )

    parser.add_argument(

        "--cup_results_dir",

        type=str,

        default=None,

        help="Directory containing cup segmentation statistical analysis results (default: Statistics/cup)"

    )

    parser.add_argument(

        "--output_dir",

        type=str,

        default="./plots",

        help="Output directory for generated plots (default: ./plots)"

    )

    return parser.parse_args()

def main():

    """Main function to generate all plots"""

    args = parse_args()

    

    print("=" * 80)

    print("CONSOLIDATED PLOT GENERATION TOOLKIT")

    print("=" * 80)

    

    # Determine repository root

    script_dir = os.path.dirname(os.path.abspath(__file__))

    repo_root = os.path.dirname(script_dir)

    

    # Set default paths if not provided

    disc_results_dir = args.disc_results_dir

    if disc_results_dir is None:

        disc_results_dir = os.path.join(repo_root, "Statistics", "disc")

    

    cup_results_dir = args.cup_results_dir

    if cup_results_dir is None:

        cup_results_dir = os.path.join(repo_root, "Statistics", "cup")

    

    # Create output directory

    output_base_dir = Path(args.output_dir)

    output_base_dir.mkdir(exist_ok=True, parents=True)

    

    print(f"\nConfiguration:")

    print(f"  Disc results: {disc_results_dir}")

    print(f"  Cup results: {cup_results_dir}")

    print(f"  Output directory: {output_base_dir}")

    

    # Load all data once

    print("\nLoading all CSV data...")

    results = load_all_results(disc_results_dir, cup_results_dir)

    print("Data loading complete!")

    

    # Generate all plots

    create_6_subplot_delta_plots(results, output_base_dir)

    create_all_local_vs_central_comparisons(results, output_base_dir)

    create_comprehensive_sk_vs_fl_plots(results, output_base_dir)

    create_fl_baseline_comparison_plots(results, output_base_dir)

    create_fl_vs_all_local_comparisons(results, output_base_dir)

    create_local_vs_central_comparison(results, output_base_dir)

    create_ood_wins_heatmaps(output_base_dir, disc_results_dir, cup_results_dir)

    

    print("\n" + "=" * 80)

    print("ALL PLOTS GENERATED SUCCESSFULLY!")

    print(f"Output directory: {output_base_dir}")

    print("=" * 80)

if __name__ == "__main__":

    main()
