#!/usr/bin/env python3
"""
South African Job Recruitment Dataset Generator
==============================================

Generates a synthetic dataset representing job recruitment in South Africa
with specific bias concerns relevant to the post-apartheid context.

Protected Attributes:
- Race: Black African, Coloured, Indian/Asian, White
- Gender: Male, Female, Non-binary
- Location: Urban vs Rural, specific provinces
- English Fluency: Native, Fluent, Intermediate, Basic
- Age: 18-65 years

Features:
- Education Level
- Years of Experience
- University Prestige (disadvantaged vs advantaged institutions)
- Skills Score
- Interview Score
- CV Quality Score
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_sa_recruitment_data(n_samples=800):
    """
    Generate synthetic South African recruitment dataset with realistic bias patterns
    """
    data = []
    
    # South African demographic distributions (approximate)
    race_distribution = {
        'Black African': 0.81,
        'Coloured': 0.09,
        'White': 0.08,
        'Indian/Asian': 0.02
    }
    
    # South African provinces with economic indicators
    provinces = {
        'Western Cape': {'urban_rate': 0.65, 'economic_advantage': 0.8},
        'Gauteng': {'urban_rate': 0.97, 'economic_advantage': 0.9},
        'KwaZulu-Natal': {'urban_rate': 0.55, 'economic_advantage': 0.6},
        'Eastern Cape': {'urban_rate': 0.38, 'economic_advantage': 0.4},
        'Limpopo': {'urban_rate': 0.13, 'economic_advantage': 0.3},
        'Mpumalanga': {'urban_rate': 0.42, 'economic_advantage': 0.5},
        'North West': {'urban_rate': 0.34, 'economic_advantage': 0.4},
        'Northern Cape': {'urban_rate': 0.70, 'economic_advantage': 0.5},
        'Free State': {'urban_rate': 0.71, 'economic_advantage': 0.5}
    }
    
    # Historically advantaged vs disadvantaged universities
    universities = {
        'advantaged': ['UCT', 'Wits', 'Stellenbosch', 'Rhodes', 'UKZN'],
        'disadvantaged': ['UWC', 'Fort Hare', 'Unisa', 'TUT', 'CPUT', 'VUT']
    }
    
    for i in range(n_samples):
        # Generate basic demographics
        race = np.random.choice(list(race_distribution.keys()), 
                               p=list(race_distribution.values()))
        gender = np.random.choice(['Male', 'Female', 'Non-binary'], 
                                 p=[0.49, 0.49, 0.02])
        age = np.random.randint(18, 66)
        
        # Generate location with bias
        province = np.random.choice(list(provinces.keys()))
        location_type = np.random.choice(['Urban', 'Rural'], 
                                        p=[provinces[province]['urban_rate'], 
                                           1 - provinces[province]['urban_rate']])
        
        # Generate English fluency with bias based on race and location
        if race == 'White':
            english_fluency = np.random.choice(['Native', 'Fluent'], p=[0.8, 0.2])
        elif race == 'Indian/Asian':
            english_fluency = np.random.choice(['Native', 'Fluent', 'Intermediate'], p=[0.3, 0.5, 0.2])
        elif location_type == 'Urban':
            english_fluency = np.random.choice(['Fluent', 'Intermediate', 'Basic'], p=[0.4, 0.4, 0.2])
        else:  # Rural
            english_fluency = np.random.choice(['Intermediate', 'Basic'], p=[0.3, 0.7])
        
        # Generate education with historical bias
        if race == 'White':
            education_level = np.random.choice(['Matric', 'Diploma', 'Bachelor', 'Honours', 'Masters', 'PhD'], 
                                             p=[0.1, 0.15, 0.3, 0.25, 0.15, 0.05])
            university_type = np.random.choice(['advantaged', 'disadvantaged'], p=[0.8, 0.2])
        elif race == 'Indian/Asian':
            education_level = np.random.choice(['Matric', 'Diploma', 'Bachelor', 'Honours', 'Masters', 'PhD'], 
                                             p=[0.15, 0.2, 0.3, 0.2, 0.12, 0.03])
            university_type = np.random.choice(['advantaged', 'disadvantaged'], p=[0.6, 0.4])
        else:  # Black African, Coloured
            education_level = np.random.choice(['Matric', 'Diploma', 'Bachelor', 'Honours', 'Masters', 'PhD'], 
                                             p=[0.4, 0.3, 0.2, 0.08, 0.015, 0.005])
            university_type = np.random.choice(['advantaged', 'disadvantaged'], p=[0.2, 0.8])
        
        # Select specific university
        if education_level in ['Bachelor', 'Honours', 'Masters', 'PhD']:
            university = np.random.choice(universities[university_type])
        else:
            university = 'None'
        
        # Generate years of experience
        years_experience = max(0, age - 22 + np.random.randint(-3, 4))
        
        # Generate skills score with bias
        base_skills = np.random.normal(70, 15)
        
        # Apply biases
        if education_level in ['Masters', 'PhD']:
            base_skills += 15
        elif education_level in ['Bachelor', 'Honours']:
            base_skills += 10
        elif education_level == 'Diploma':
            base_skills += 5
        
        if university_type == 'advantaged':
            base_skills += 8
        
        if english_fluency == 'Native':
            base_skills += 10
        elif english_fluency == 'Fluent':
            base_skills += 5
        elif english_fluency == 'Basic':
            base_skills -= 10
        
        if location_type == 'Urban':
            base_skills += 5
        
        skills_score = np.clip(base_skills, 0, 100)
        
        # Generate interview score with bias
        base_interview = np.random.normal(65, 12)
        
        # Language bias in interviews
        if english_fluency == 'Native':
            base_interview += 15
        elif english_fluency == 'Fluent':
            base_interview += 8
        elif english_fluency == 'Intermediate':
            base_interview += 2
        else:  # Basic
            base_interview -= 10
        
        # Racial bias in interviews (unconscious bias)
        if race == 'White':
            base_interview += 8
        elif race == 'Indian/Asian':
            base_interview += 3
        elif race == 'Coloured':
            base_interview -= 2
        else:  # Black African
            base_interview -= 5
        
        # Gender bias
        if gender == 'Male':
            base_interview += 3
        elif gender == 'Non-binary':
            base_interview -= 8
        
        # Age bias
        if age > 50:
            base_interview -= 8
        elif age < 25:
            base_interview -= 3
        
        interview_score = np.clip(base_interview, 0, 100)
        
        # Generate CV quality score
        base_cv = np.random.normal(60, 10)
        
        if university_type == 'advantaged':
            base_cv += 12
        
        if location_type == 'Urban':
            base_cv += 5
        
        if english_fluency in ['Native', 'Fluent']:
            base_cv += 8
        
        cv_score = np.clip(base_cv, 0, 100)
        
        # Generate true hiring suitability (unbiased measure)
        true_suitability = (skills_score * 0.4 + 
                          (years_experience * 2) * 0.3 + 
                          (1 if education_level in ['Bachelor', 'Honours', 'Masters', 'PhD'] else 0) * 20 * 0.3)
        
        true_label = 1 if true_suitability > 60 else 0
        
        # Generate biased model prediction
        biased_score = (cv_score * 0.3 + 
                       interview_score * 0.4 + 
                       skills_score * 0.3)
        
        # Add systematic bias
        if race == 'White':
            biased_score += 8
        elif race == 'Indian/Asian':
            biased_score += 4
        elif race == 'Black African':
            biased_score -= 6
        elif race == 'Coloured':
            biased_score -= 3
        
        if gender == 'Male':
            biased_score += 3
        elif gender == 'Non-binary':
            biased_score -= 10
        
        if english_fluency == 'Native':
            biased_score += 10
        elif english_fluency == 'Basic':
            biased_score -= 8
        
        if location_type == 'Rural':
            biased_score -= 5
        
        model_prediction = 1 if biased_score > 65 else 0
        
        # Create record
        record = {
            'age': age,
            'gender': gender,
            'race': race,
            'province': province,
            'location_type': location_type,
            'english_fluency': english_fluency,
            'education_level': education_level,
            'university': university,
            'university_type': university_type,
            'years_experience': years_experience,
            'skills_score': round(skills_score, 1),
            'interview_score': round(interview_score, 1),
            'cv_score': round(cv_score, 1),
            'true_suitability_score': round(true_suitability, 1),
            'true_label': true_label,
            'model_prediction': model_prediction
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

def main():
    """Generate and save the South African recruitment dataset"""
    print("🇿🇦 Generating South African Job Recruitment Dataset...")
    
    # Generate dataset
    df = generate_sa_recruitment_data(800)
    
    # Save to CSV
    df.to_csv('sa_recruitment_data.csv', index=False)
    
    print(f"✅ Dataset generated successfully!")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"💾 Saved as: sa_recruitment_data.csv")
    
    # Display basic statistics
    print("\n📈 DATASET OVERVIEW:")
    print("="*50)
    
    print(f"Total Records: {len(df):,}")
    print(f"Positive Rate (True): {df['true_label'].mean():.2%}")
    print(f"Positive Rate (Model): {df['model_prediction'].mean():.2%}")
    
    print("\n🏛️ DEMOGRAPHIC BREAKDOWN:")
    print("-"*30)
    
    for attr in ['race', 'gender', 'location_type', 'english_fluency']:
        print(f"\n{attr.upper()}:")
        counts = df[attr].value_counts()
        for category, count in counts.items():
            pct = (count / len(df)) * 100
            print(f"  {category}: {count} ({pct:.1f}%)")
    
    print("\n🎯 BIAS INDICATORS:")
    print("-"*30)
    
    # Quick bias check
    for attr in ['race', 'gender', 'location_type', 'english_fluency']:
        group_rates = df.groupby(attr)['model_prediction'].mean()
        max_rate = group_rates.max()
        min_rate = group_rates.min()
        bias_gap = max_rate - min_rate
        
        print(f"{attr}: {bias_gap:.3f} gap (Max: {max_rate:.3f}, Min: {min_rate:.3f})")
        if bias_gap > 0.1:
            print(f"  ⚠️  Significant bias detected in {attr}!")

if __name__ == "__main__":
    main() 