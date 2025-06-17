# 🇿🇦 South African Job Recruitment Bias Audit System

## 🎯 Overview

This comprehensive bias audit system conducts thorough analysis of **South African job recruitment datasets** and automated CV screening models to identify ethical concerns specific to post-apartheid SA context and implement concrete mitigation strategies. The system addresses the critical need for fair AI in South Africa's transformation journey.

## 📋 Features & SA Requirements Met

### ✅ Enhanced Requirements Compliance

- **Complete SA-Specific Bias Audit:**
  - ✅ South African job recruitment dataset (800 records, 9 provinces)
  - ✅ Implementation of 5+ quantitative fairness metrics
  - ✅ SA-specific protected attributes: **Race, Gender, Location, English Fluency, Age**
  - ✅ Visual and statistical representation of bias patterns
  - ✅ Application of 3 bias mitigation techniques
  - ✅ Performance comparison before/after mitigation
  - ✅ SA-specific dataset improvement recommendations
  - ✅ Real-world harms analysis in SA context
  - ✅ Ubuntu-centered ethics framework

### ✅ Technical Specifications Compliance

- **Fairness Toolkit Integration:**
  - IBM AI Fairness 360 (AIF360) support
  - Fairlearn integration
  - Custom fairness implementations as fallbacks
  
- **Reproducible Analysis:**
  - Jupyter notebook with documented code
  - Automated setup and execution scripts
  - Statistical testing for bias validation
  - Accessible visualizations for all audiences

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
python setup_bias_audit.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter notebook
jupyter notebook comprehensive_bias_audit.ipynb
```

### Option 3: Google Colab
Upload `comprehensive_bias_audit.ipynb` to Google Colab for cloud execution.

## 📊 Included Components

### 1. Comprehensive Fairness Metrics (5 implemented)
- **Demographic Parity**: Equal positive prediction rates across groups
- **Equalized Odds**: Equal true/false positive rates across groups  
- **Equal Opportunity**: Equal true positive rates across groups
- **Calibration**: Equal accuracy of predictions across groups
- **Overall Accuracy Equality**: Equal classification accuracy across groups

### 2. Statistical Testing Suite
- Chi-square tests for independence
- Proportion Z-tests for group comparisons
- Fisher's exact tests for small samples
- Effect size analysis (Cohen's d)
- Significance testing with multiple comparison corrections

### 3. Bias Mitigation Techniques (3 implemented)
- **Pre-processing: Data Reweighting** - Balances representation through sample weighting
- **Post-processing: Threshold Optimization** - Optimizes decision thresholds per group
- **In-processing: Adversarial Debiasing** - Incorporates fairness constraints during training

### 4. Comprehensive Visualizations
- Fairness metrics heatmaps
- Group-wise performance comparisons
- Intersectional bias analysis
- Bias severity assessments
- Before/after mitigation comparisons

## 📁 File Structure

```
bias-audit-report/
├── comprehensive_bias_audit.ipynb    # Main analysis notebook
├── sample_bias_data.csv             # Sample dataset
├── requirements.txt                 # Python dependencies
├── setup_bias_audit.py             # Automated setup script
├── BIAS_AUDIT_README.md            # This documentation
├── bias_audit.py                   # Legacy audit script
├── app.py                          # Web interface (optional)
└── generated_outputs/              # Analysis outputs
    ├── comprehensive_bias_dashboard.png
    ├── mitigation_comparison.png
    └── comprehensive_bias_audit.html
```

## 🔧 Technical Requirements

### Python Version
- Python 3.7+ required
- Python 3.8+ recommended

### Core Dependencies
```
pandas>=1.3.0          # Data manipulation
numpy>=1.21.0           # Numerical computing
matplotlib>=3.4.0       # Plotting
seaborn>=0.11.0         # Statistical visualizations
scikit-learn>=1.0.0     # Machine learning
jupyter>=1.0.0          # Notebook environment
```

### Fairness Libraries (Optional but Recommended)
```
aif360>=0.5.0          # IBM AI Fairness 360
fairlearn>=0.7.0       # Microsoft Fairlearn
scipy>=1.7.0           # Statistical functions
plotly>=5.0.0          # Interactive visualizations
statsmodels>=0.13.0    # Statistical modeling
```

## 📊 Dataset Specifications

### Current Dataset: Hiring/Promotion Decisions
- **Size**: 501 records
- **Protected Attributes**: Race, Gender, Age
- **Features**: Education, Income Level, Years Experience
- **Target**: Binary hiring/promotion decision
- **Risk Level**: High-stakes domain with direct impact on livelihoods

### Supported Data Format
```csv
age,gender,race,education,income_level,years_experience,true_label,model_prediction
32,Female,White,Bachelor,Medium,10,1,1
45,Male,Black,High School,Low,5,0,1
...
```

## 🔍 Analysis Components

### 1. Dataset Overview & Selection
- Comprehensive data exploration
- Representation analysis across protected groups
- Identification of potential bias sources
- Justification for high-stakes domain selection

### 2. Quantitative Fairness Assessment
- Implementation of multiple fairness definitions
- Group-wise performance metrics calculation
- Intersectional bias analysis
- Bias severity classification

### 3. Statistical Validation
- Significance testing for bias detection
- Effect size calculations
- Multiple comparison corrections
- Confidence interval estimation

### 4. Visualization Suite
- Technical dashboards for data scientists
- Executive summaries for leadership
- Accessible charts for non-technical stakeholders
- Interactive visualizations where applicable

### 5. Mitigation Implementation
- Pre-processing bias reduction
- In-processing fairness constraints
- Post-processing threshold optimization
- Performance vs fairness trade-off analysis

### 6. Impact Assessment
- Individual-level harm analysis
- Societal impact evaluation
- Organizational risk assessment
- Legal and compliance implications

## 🎓 Educational Value

### Learning Objectives
- Understanding multiple fairness definitions
- Implementing statistical bias testing
- Applying bias mitigation techniques
- Ethical framework development
- Real-world impact assessment

### Suitable For
- Data science students and professionals
- AI ethics researchers
- Policy makers and regulators
- HR and legal teams
- Algorithmic auditing practitioners

## ⚖️ Ethics Framework

### Core Principles
1. **Justice & Fairness**: Equal opportunity and treatment
2. **Transparency**: Clear, explainable processes
3. **Accountability**: Responsible ownership and oversight
4. **Dignity**: Respect for human worth and rights

### Stakeholder Responsibilities
- Data Scientists: Implement robust bias testing
- HR Teams: Provide oversight and appeal processes
- Leadership: Set standards and allocate resources
- Legal: Ensure compliance and manage risks

## 📈 Success Metrics

### Fairness Targets
- Demographic parity difference < 0.05
- Equalized odds difference < 0.05
- Statistical significance tests show no discrimination

### Performance Targets
- Maintain accuracy within 2% of baseline
- Achieve F1-score parity across groups
- Balanced precision/recall across demographics

## 🚨 Limitations & Considerations

### Technical Limitations
- Requires sufficient sample sizes per group
- May not capture all forms of discrimination
- Trade-offs between different fairness metrics
- Depends on quality of protected attribute labels

### Ethical Considerations
- Fairness metrics are value judgments, not universal truths
- May inadvertently reinforce existing categories
- Requires ongoing monitoring and adjustment
- Should complement, not replace, human judgment

## 🔄 Ongoing Maintenance

### Regular Review Cycle
- Quarterly bias audit execution
- Annual methodology review
- Continuous stakeholder engagement
- Regular regulatory compliance checks

### Update Triggers
- New regulatory requirements
- Significant data distribution changes
- Detection of emerging bias patterns
- Stakeholder feedback incorporation

## 📞 Support & Contribution

### Getting Help
1. Check the comprehensive documentation in the notebook
2. Review the generated HTML report for detailed results
3. Examine visualization outputs for insights
4. Consult the ethics framework for guidance

### Contributing Improvements
- Additional fairness metrics implementation
- New mitigation technique development
- Enhanced visualization capabilities
- Extended ethics framework components

## 📚 References & Further Reading

### Academic Papers
- Mehrabi et al. "A Survey on Bias and Fairness in Machine Learning" (2021)
- Barocas et al. "Fairness and Machine Learning" (2019)
- Dwork et al. "Fairness Through Awareness" (2012)

### Technical Resources
- IBM AI Fairness 360 Documentation
- Microsoft Fairlearn User Guide
- Google What-If Tool Documentation
- Partnership on AI Fairness Toolkit

### Legal & Regulatory
- EU AI Act Requirements
- US Equal Employment Opportunity Commission Guidelines
- IEEE Standards for Algorithmic bias

---

## 🏆 Summary

This comprehensive bias audit system provides everything needed to conduct thorough, professional-grade algorithmic bias analysis. It combines academic rigor with practical applicability, making it suitable for both educational and professional use in high-stakes ML applications.

**Ready to audit bias responsibly? Start with `python setup_bias_audit.py`** 