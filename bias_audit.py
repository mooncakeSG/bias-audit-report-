#!/usr/bin/env python3
"""
Bias Audit Tool for Machine Learning Models
===========================================

This script performs comprehensive bias auditing for machine learning models
by analyzing fairness metrics across different demographic groups.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class BiasAuditor:
    """
    A comprehensive bias auditing tool for machine learning models.
    """
    
    def __init__(self, data, protected_attributes, target_column, prediction_column):
        """
        Initialize the BiasAuditor with data and column specifications.
        
        Args:
            data (pd.DataFrame): Dataset containing predictions and true labels
            protected_attributes (list): List of protected attribute column names
            target_column (str): Name of the true label column
            prediction_column (str): Name of the prediction column
        """
        self.data = data.copy()
        self.protected_attributes = protected_attributes
        self.target_column = target_column
        self.prediction_column = prediction_column
        self.results = {}
        
    def calculate_fairness_metrics(self):
        """
        Calculate various fairness metrics for each protected attribute.
        """
        results = []
        
        for attr in self.protected_attributes:
            # Get unique groups for this attribute
            groups = self.data[attr].unique()
            
            for group in groups:
                group_data = self.data[self.data[attr] == group]
                
                if len(group_data) == 0:
                    continue
                    
                # Calculate basic metrics
                y_true = group_data[self.target_column]
                y_pred = group_data[self.prediction_column]
                
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                # Calculate fairness-specific metrics
                positive_rate = np.mean(y_pred == 1) if len(np.unique(y_pred)) > 1 else 0
                true_positive_rate = recall
                false_positive_rate = self._calculate_fpr(y_true, y_pred)
                
                results.append({
                    'protected_attribute': attr,
                    'group': group,
                    'sample_size': len(group_data),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'positive_prediction_rate': positive_rate,
                    'true_positive_rate': true_positive_rate,
                    'false_positive_rate': false_positive_rate
                })
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def _calculate_fpr(self, y_true, y_pred):
        """Calculate False Positive Rate."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0
    
    def calculate_bias_metrics(self):
        """
        Calculate bias metrics comparing different groups.
        """
        bias_results = []
        
        for attr in self.protected_attributes:
            attr_data = self.results[self.results['protected_attribute'] == attr]
            
            if len(attr_data) < 2:
                continue
                
            # Calculate demographic parity difference
            max_ppr = attr_data['positive_prediction_rate'].max()
            min_ppr = attr_data['positive_prediction_rate'].min()
            demographic_parity_diff = max_ppr - min_ppr
            
            # Calculate equalized odds difference
            max_tpr = attr_data['true_positive_rate'].max()
            min_tpr = attr_data['true_positive_rate'].min()
            tpr_diff = max_tpr - min_tpr
            
            max_fpr = attr_data['false_positive_rate'].max()
            min_fpr = attr_data['false_positive_rate'].min()
            fpr_diff = max_fpr - min_fpr
            
            equalized_odds_diff = max(tpr_diff, fpr_diff)
            
            # Calculate accuracy difference
            max_acc = attr_data['accuracy'].max()
            min_acc = attr_data['accuracy'].min()
            accuracy_diff = max_acc - min_acc
            
            bias_results.append({
                'protected_attribute': attr,
                'demographic_parity_difference': demographic_parity_diff,
                'equalized_odds_difference': equalized_odds_diff,
                'accuracy_difference': accuracy_diff,
                'bias_severity': self._assess_bias_severity(demographic_parity_diff, equalized_odds_diff)
            })
        
        return pd.DataFrame(bias_results)
    
    def _assess_bias_severity(self, dp_diff, eo_diff):
        """Assess the severity of bias based on fairness metrics."""
        max_diff = max(dp_diff, eo_diff)
        
        if max_diff < 0.05:
            return 'Low'
        elif max_diff < 0.1:
            return 'Moderate'
        elif max_diff < 0.2:
            return 'High'
        else:
            return 'Severe'
    
    def generate_visualizations(self, save_path='bias_audit_chart.png'):
        """
        Generate bias audit visualizations.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bias Audit Report - Fairness Metrics Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy by Protected Groups
        for i, attr in enumerate(self.protected_attributes):
            if i >= 2:  # Limit to 2 attributes for visualization
                break
                
            attr_data = self.results[self.results['protected_attribute'] == attr]
            
            ax = axes[0, i]
            
            sns.barplot(data=attr_data, x='group', y='accuracy', ax=ax)
            ax.set_title(f'Accuracy by {attr}')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
        
        # Plot 2: Positive Prediction Rate by Groups
        sns.boxplot(data=self.results, x='protected_attribute', y='positive_prediction_rate', ax=axes[1, 0])
        axes[1, 0].set_title('Positive Prediction Rate Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 3: Bias Severity Assessment
        bias_df = self.calculate_bias_metrics()
        severity_counts = bias_df['bias_severity'].value_counts()
        axes[1, 1].pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Bias Severity Distribution')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Bias audit visualization saved to: {save_path}")
    
    def save_results(self, csv_path='bias_audit_results.csv'):
        """
        Save detailed results to CSV file.
        """
        if self.results is not None and not self.results.empty:
            # Save detailed results
            self.results.to_csv(csv_path, index=False)
            
            # Save bias summary
            bias_metrics = self.calculate_bias_metrics()
            bias_summary_path = csv_path.replace('.csv', '_summary.csv')
            bias_metrics.to_csv(bias_summary_path, index=False)
            
            print(f"Detailed results saved to: {csv_path}")
            print(f"Bias summary saved to: {bias_summary_path}")
        else:
            print("No results to save. Please run calculate_fairness_metrics() first.")

def generate_sample_data():
    """
    Generate sample data for demonstration purposes.
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic data with potential bias
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
        'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Generate target variable with some bias
    df['true_label'] = 0
    for idx, row in df.iterrows():
        prob = 0.3  # Base probability
        
        # Introduce bias
        if row['gender'] == 'Male':
            prob += 0.2
        if row['race'] == 'White':
            prob += 0.15
        if row['education'] in ['Master', 'PhD']:
            prob += 0.25
        if row['age'] > 40:
            prob += 0.1
            
        df.loc[idx, 'true_label'] = np.random.binomial(1, min(prob, 0.9))
    
    # Generate predictions with similar bias pattern
    df['predictions'] = 0
    for idx, row in df.iterrows():
        prob = 0.35  # Base probability
        
        # Model exhibits bias
        if row['gender'] == 'Male':
            prob += 0.25
        if row['race'] == 'White':
            prob += 0.20
        if row['education'] in ['Master', 'PhD']:
            prob += 0.20
        if row['age'] > 40:
            prob += 0.05
            
        df.loc[idx, 'predictions'] = np.random.binomial(1, min(prob, 0.95))
    
    return df

def main():
    """
    Main function to run the bias audit.
    """
    print("Starting Bias Audit Analysis...")
    print("=" * 50)
    
    # Generate or load data
    print("Loading data...")
    data = generate_sample_data()
    print(f"Data loaded: {len(data)} samples")
    
    # Initialize bias auditor
    protected_attributes = ['gender', 'race', 'education']
    auditor = BiasAuditor(
        data=data,
        protected_attributes=protected_attributes,
        target_column='true_label',
        prediction_column='predictions'
    )
    
    # Calculate fairness metrics
    print("\nCalculating fairness metrics...")
    results = auditor.calculate_fairness_metrics()
    print(f"Fairness metrics calculated for {len(results)} groups")
    
    # Calculate bias metrics
    print("\nCalculating bias metrics...")
    bias_metrics = auditor.calculate_bias_metrics()
    print("Bias metrics calculated")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    auditor.generate_visualizations()
    
    # Save results
    print("\nSaving results...")
    auditor.save_results()
    
    # Print summary
    print("\n" + "=" * 50)
    print("BIAS AUDIT SUMMARY")
    print("=" * 50)
    
    for _, row in bias_metrics.iterrows():
        print(f"\n{row['protected_attribute'].upper()}:")
        print(f"  Demographic Parity Difference: {row['demographic_parity_difference']:.3f}")
        print(f"  Equalized Odds Difference: {row['equalized_odds_difference']:.3f}")
        print(f"  Accuracy Difference: {row['accuracy_difference']:.3f}")
        print(f"  Bias Severity: {row['bias_severity']}")
    
    print("\nBias audit completed successfully!")
    print("Check the generated files for detailed results.")

if __name__ == "__main__":
    main() 