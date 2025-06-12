#!/usr/bin/env python3
"""
Generate Sample Data for Bias Audit Testing
===========================================

This script creates a realistic CSV dataset with intentional bias patterns
for testing the bias audit dashboard.
"""

import pandas as pd
import numpy as np

def generate_sample_data():
    """Generate sample data with bias patterns."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 500
    
    # Create demographic data
    data = {
        'age': np.random.randint(22, 65, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45]),
        'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], n_samples, p=[0.6, 0.2, 0.12, 0.06, 0.02]),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.35, 0.4, 0.2, 0.05]),
        'income_level': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2]),
        'years_experience': np.random.randint(0, 25, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate true labels with realistic patterns
    df['true_label'] = 0
    for idx, row in df.iterrows():
        prob = 0.4  # Base probability
        
        # Education influence
        if row['education'] == 'PhD':
            prob += 0.3
        elif row['education'] == 'Master':
            prob += 0.2
        elif row['education'] == 'Bachelor':
            prob += 0.1
        
        # Experience influence
        if row['years_experience'] > 10:
            prob += 0.15
        elif row['years_experience'] > 5:
            prob += 0.1
        
        # Income influence
        if row['income_level'] == 'High':
            prob += 0.2
        elif row['income_level'] == 'Medium':
            prob += 0.1
        
        df.loc[idx, 'true_label'] = np.random.binomial(1, min(prob, 0.95))
    
    # Generate biased predictions
    df['model_prediction'] = 0
    for idx, row in df.iterrows():
        prob = 0.35  # Base probability
        
        # Education influence (similar to true labels)
        if row['education'] == 'PhD':
            prob += 0.25
        elif row['education'] == 'Master':
            prob += 0.18
        elif row['education'] == 'Bachelor':
            prob += 0.08
        
        # Experience influence
        if row['years_experience'] > 10:
            prob += 0.12
        elif row['years_experience'] > 5:
            prob += 0.08
        
        # BIAS: Gender bias (favoring males)
        if row['gender'] == 'Male':
            prob += 0.15
        
        # BIAS: Racial bias (favoring certain groups)
        if row['race'] == 'White':
            prob += 0.12
        elif row['race'] == 'Asian':
            prob += 0.08
        
        # Income influence
        if row['income_level'] == 'High':
            prob += 0.18
        elif row['income_level'] == 'Medium':
            prob += 0.08
        
        df.loc[idx, 'model_prediction'] = np.random.binomial(1, min(prob, 0.95))
    
    return df

if __name__ == '__main__':
    # Generate the sample data
    df = generate_sample_data()
    
    # Save to CSV
    filename = 'sample_bias_data.csv'
    df.to_csv(filename, index=False)
    
    # Print summary
    print(f"✅ Sample data saved to '{filename}'")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"🎯 True label distribution: {df['true_label'].value_counts().to_dict()}")
    print(f"🤖 Prediction distribution: {df['model_prediction'].value_counts().to_dict()}")
    
    print("\n📈 Sample Data Preview:")
    print(df.head(10).to_string(index=False))
    
    print("\n🔍 Bias Patterns Included:")
    print("- Gender bias: Males favored in predictions")
    print("- Racial bias: White and Asian groups favored")
    print("- Education correlation: Higher education = higher prediction rates")
    print("- Experience correlation: More experience = higher rates")
    
    print(f"\n🚀 To test the dashboard:")
    print(f"1. Open http://localhost:5000")
    print(f"2. Upload '{filename}'")
    print("3. Configure:")
    print("   - Target Column: true_label")
    print("   - Prediction Column: model_prediction") 
    print("   - Protected Attributes: gender, race, education")
    print("4. Run Bias Audit and explore results!") 