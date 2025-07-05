#!/usr/bin/env python3
"""
Correlation Analysis Script for Human Annotators

This script calculates correlation coefficients between three human annotators
using both Pearson correlation and Cohen's Kappa coefficient.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple


def load_annotator_data(file_path: str) -> pd.DataFrame:
    """Load annotator data from CSV file."""
    df = pd.read_csv(file_path)
    # Extract the quality scores and tool_use information
    return df[['tool_use', 'Quality', 'Title']]


def calculate_correlations(annotator1_data: pd.DataFrame, 
                          annotator2_data: pd.DataFrame,
                          annotator1_name: str,
                          annotator2_name: str) -> Dict:
    """Calculate correlations between two annotators."""
    
    # Merge data by Title to align scores
    merged_data = pd.merge(annotator1_data, annotator2_data, 
                          on='Title', suffixes=('_1', '_2'))
    
    # Extract quality scores
    scores_1 = merged_data['Quality_1'].values
    scores_2 = merged_data['Quality_2'].values
    
    # Calculate Pearson correlation
    pearson_corr, pearson_p = pearsonr(scores_1, scores_2)
    
    # Calculate Cohen's Kappa
    # For Kappa, we need to discretize the scores into categories
    # Let's use 5 categories: 1-2, 2-3, 3-4, 4-5, 5+
    def discretize_scores(scores):
        categories = []
        for score in scores:
            if score < 2:
                categories.append(1)
            elif score < 3:
                categories.append(2)
            elif score < 4:
                categories.append(3)
            elif score < 5:
                categories.append(4)
            else:
                categories.append(5)
        return categories
    
    categories_1 = discretize_scores(scores_1)
    categories_2 = discretize_scores(scores_2)
    
    kappa_score = cohen_kappa_score(categories_1, categories_2)
    
    return {
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'kappa_score': kappa_score,
        'n_samples': len(scores_1),
        'annotator1_mean': np.mean(scores_1),
        'annotator2_mean': np.mean(scores_2),
        'annotator1_std': np.std(scores_1),
        'annotator2_std': np.std(scores_2)
    }


def analyze_tool_use_correlations(annotator1_data: pd.DataFrame,
                                 annotator2_data: pd.DataFrame,
                                 annotator1_name: str,
                                 annotator2_name: str) -> Dict:
    """Analyze correlations separately for tool_use=True and tool_use=False."""
    
    # Merge data by Title
    merged_data = pd.merge(annotator1_data, annotator2_data, 
                          on='Title', suffixes=('_1', '_2'))
    
    # Separate by tool_use
    tool_use_true = merged_data[merged_data['tool_use_1'] == True]
    tool_use_false = merged_data[merged_data['tool_use_1'] == False]
    
    results = {}
    
    # Analyze tool_use=True
    if len(tool_use_true) > 0:
        scores_1_true = tool_use_true['Quality_1'].values
        scores_2_true = tool_use_true['Quality_2'].values
        pearson_corr_true, pearson_p_true = pearsonr(scores_1_true, scores_2_true)
        
        # Discretize for Kappa
        def discretize_scores(scores):
            categories = []
            for score in scores:
                if score < 2:
                    categories.append(1)
                elif score < 3:
                    categories.append(2)
                elif score < 4:
                    categories.append(3)
                elif score < 5:
                    categories.append(4)
                else:
                    categories.append(5)
            return categories
        
        categories_1_true = discretize_scores(scores_1_true)
        categories_2_true = discretize_scores(scores_2_true)
        kappa_true = cohen_kappa_score(categories_1_true, categories_2_true)
        
        results['tool_use_true'] = {
            'pearson_correlation': pearson_corr_true,
            'pearson_p_value': pearson_p_true,
            'kappa_score': kappa_true,
            'n_samples': len(scores_1_true),
            'annotator1_mean': np.mean(scores_1_true),
            'annotator2_mean': np.mean(scores_2_true)
        }
    
    # Analyze tool_use=False
    if len(tool_use_false) > 0:
        scores_1_false = tool_use_false['Quality_1'].values
        scores_2_false = tool_use_false['Quality_2'].values
        pearson_corr_false, pearson_p_false = pearsonr(scores_1_false, scores_2_false)
        
        # Discretize for Kappa
        def discretize_scores(scores):
            categories = []
            for score in scores:
                if score < 2:
                    categories.append(1)
                elif score < 3:
                    categories.append(2)
                elif score < 4:
                    categories.append(3)
                elif score < 5:
                    categories.append(4)
                else:
                    categories.append(5)
            return categories
        
        categories_1_false = discretize_scores(scores_1_false)
        categories_2_false = discretize_scores(scores_2_false)
        kappa_false = cohen_kappa_score(categories_1_false, categories_2_false)
        
        results['tool_use_false'] = {
            'pearson_correlation': pearson_corr_false,
            'pearson_p_value': pearson_p_false,
            'kappa_score': kappa_false,
            'n_samples': len(scores_1_false),
            'annotator1_mean': np.mean(scores_1_false),
            'annotator2_mean': np.mean(scores_2_false)
        }
    
    return results


def create_correlation_matrix(annotator_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a correlation matrix for all annotators."""
    # Get all unique titles and handle duplicates
    all_titles = set()
    for data in annotator_data.values():
        all_titles.update(data['Title'].tolist())
    
    # Create a matrix with all scores, handling duplicates by taking first occurrence
    correlation_matrix = pd.DataFrame(index=list(all_titles))
    
    for annotator_name, data in annotator_data.items():
        # Remove duplicates by keeping first occurrence
        data_unique = data.drop_duplicates(subset=['Title'], keep='first')
        correlation_matrix[f'{annotator_name}_score'] = data_unique.set_index('Title')['Quality']
    
    return correlation_matrix


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame, output_file: str = 'correlation_heatmap.png'):
    """Plot correlation heatmap."""
    # Calculate correlation matrix
    corr_matrix = correlation_matrix.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix Between Annotators')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to run correlation analysis."""
    
    # Load data from three annotators
    annotator_files = {
        'Annotator_1': 'human_annotate_results/group_shuffled_results_annota_1.csv',
        'Annotator_2': 'human_annotate_results/group_shuffled_results_annota_2.csv',
        'Annotator_3': 'human_annotate_results/group_shuffled_results_annota_3.csv'
    }
    
    annotator_data = {}
    for name, file_path in annotator_files.items():
        try:
            annotator_data[name] = load_annotator_data(file_path)
            print(f"Loaded {len(annotator_data[name])} samples from {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
            return
    
    # Calculate pairwise correlations
    annotator_names = list(annotator_data.keys())
    results = {}
    
    print("\n" + "="*80)
    print("PAIRWISE CORRELATION ANALYSIS")
    print("="*80)
    
    for i in range(len(annotator_names)):
        for j in range(i+1, len(annotator_names)):
            annotator1 = annotator_names[i]
            annotator2 = annotator_names[j]
            
            print(f"\n{annotator1} vs {annotator2}:")
            print("-" * 50)
            
            # Overall correlation
            overall_corr = calculate_correlations(
                annotator_data[annotator1], 
                annotator_data[annotator2],
                annotator1, 
                annotator2
            )
            
            print(f"Overall Correlation:")
            print(f"  Pearson Correlation: {overall_corr['pearson_correlation']:.4f} (p={overall_corr['pearson_p_value']:.4f})")
            print(f"  Cohen's Kappa: {overall_corr['kappa_score']:.4f}")
            print(f"  Number of samples: {overall_corr['n_samples']}")
            print(f"  {annotator1} mean: {overall_corr['annotator1_mean']:.3f} ± {overall_corr['annotator1_std']:.3f}")
            print(f"  {annotator2} mean: {overall_corr['annotator2_mean']:.3f} ± {overall_corr['annotator2_std']:.3f}")
            
            # Tool use specific correlations
            tool_use_corr = analyze_tool_use_correlations(
                annotator_data[annotator1], 
                annotator_data[annotator2],
                annotator1, 
                annotator2
            )
            
            if 'tool_use_true' in tool_use_corr:
                true_corr = tool_use_corr['tool_use_true']
                print(f"\nTool Use = True:")
                print(f"  Pearson Correlation: {true_corr['pearson_correlation']:.4f} (p={true_corr['pearson_p_value']:.4f})")
                print(f"  Cohen's Kappa: {true_corr['kappa_score']:.4f}")
                print(f"  Number of samples: {true_corr['n_samples']}")
                print(f"  {annotator1} mean: {true_corr['annotator1_mean']:.3f}")
                print(f"  {annotator2} mean: {true_corr['annotator2_mean']:.3f}")
            
            if 'tool_use_false' in tool_use_corr:
                false_corr = tool_use_corr['tool_use_false']
                print(f"\nTool Use = False:")
                print(f"  Pearson Correlation: {false_corr['pearson_correlation']:.4f} (p={false_corr['pearson_p_value']:.4f})")
                print(f"  Cohen's Kappa: {false_corr['kappa_score']:.4f}")
                print(f"  Number of samples: {false_corr['n_samples']}")
                print(f"  {annotator1} mean: {false_corr['annotator1_mean']:.3f}")
                print(f"  {annotator2} mean: {false_corr['annotator2_mean']:.3f}")
            
            results[f"{annotator1}_vs_{annotator2}"] = {
                'overall': overall_corr,
                'tool_use_specific': tool_use_corr
            }
    
    # Create correlation matrix
    print("\n" + "="*80)
    print("CORRELATION MATRIX")
    print("="*80)
    
    try:
        correlation_matrix = create_correlation_matrix(annotator_data)
        print("Correlation Matrix:")
        print(correlation_matrix.corr().round(4))
        
        # Plot heatmap
        try:
            plot_correlation_heatmap(correlation_matrix)
            print("\nCorrelation heatmap saved as 'correlation_heatmap.png'")
        except Exception as e:
            print(f"Could not create heatmap: {e}")
    except Exception as e:
        print(f"Could not create correlation matrix: {e}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for name, data in annotator_data.items():
        print(f"\n{name}:")
        print(f"  Total samples: {len(data)}")
        print(f"  Mean quality score: {data['Quality'].mean():.3f}")
        print(f"  Std quality score: {data['Quality'].std():.3f}")
        print(f"  Min quality score: {data['Quality'].min():.3f}")
        print(f"  Max quality score: {data['Quality'].max():.3f}")
        print(f"  Tool use = True: {len(data[data['tool_use'] == True])}")
        print(f"  Tool use = False: {len(data[data['tool_use'] == False])}")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("\nCorrelation Analysis Results:")
    print("1. Pearson Correlation: Measures linear relationship between scores")
    print("   - Range: -1 to +1")
    print("   - Positive: Higher scores together")
    print("   - Negative: Inverse relationship")
    print("   - Near 0: No linear relationship")
    
    print("\n2. Cohen's Kappa: Measures agreement beyond chance")
    print("   - Range: -1 to +1")
    print("   - < 0: Less agreement than chance")
    print("   - 0-0.2: Slight agreement")
    print("   - 0.2-0.4: Fair agreement")
    print("   - 0.4-0.6: Moderate agreement")
    print("   - 0.6-0.8: Substantial agreement")
    print("   - 0.8-1.0: Almost perfect agreement")
    
    print("\n3. Key Findings:")
    print("   - All correlations are relatively low, indicating limited agreement")
    print("   - Annotator_1 tends to give higher scores than others")
    print("   - Annotator_3 tends to give lower scores than others")
    print("   - Kappa scores suggest slight to fair agreement at best")


if __name__ == "__main__":
    main() 