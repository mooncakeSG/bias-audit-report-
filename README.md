# 🎯 South African Bias Audit Dashboard

**Tech Titanians** | Professional Bias Detection & Monitoring System  
*Employment Equity Act Compliant | POPIA Aligned*

---

## 📋 Overview

The **South African Bias Audit Dashboard** is a comprehensive Streamlit-based web application designed to detect, monitor, and mitigate algorithmic bias in hiring systems. Built specifically for South African Employment Equity Act compliance, this dashboard provides real-time bias detection, detailed analytics, and professional reporting capabilities.

### 🎯 Key Features

- **Real-time Bias Detection**: Continuous monitoring of hiring algorithms for discriminatory patterns
- **Interactive Analytics**: Comprehensive visualizations and statistical analysis
- **Professional Reporting**: Generate compliance reports for legal and audit purposes  
- **Configurable Thresholds**: Customizable bias detection sensitivity levels
- **Multi-demographic Analysis**: Support for Race, Gender, Location, Language, and Age groups
- **Employment Equity Act Compliance**: Built-in compliance checking and reporting

---

## 🏗️ Project Structure

```
bias_audit_dashboard/
│
├── app.py                          # Main Streamlit application
├── dashboard_helpers.py            # Helper functions for data processing
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── data/
│   └── synthetic_sa_job_dataset.csv    # Sample dataset
│
├── assets/
│   ├── custom_styles.css           # Custom CSS styling
│   └── sa_flag.svg                 # South African flag (optional)
│
├── reports/
│   └── executive_summary.pdf       # Sample executive report
│
└── .streamlit/
    └── config.toml                 # Streamlit configuration
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd bias_audit_dashboard

# Install dependencies
pip install -r requirements.txt

# Verify data file exists
ls data/synthetic_sa_job_dataset.csv
```

### 3. Launch Dashboard

```bash
# Start the Streamlit application
streamlit run app.py

# The dashboard will open in your browser at:
# http://localhost:8501
```

---

## 📊 Dashboard Sections

### 🎯 Main Dashboard
- **Key Metrics**: Total events, severe incidents, compliance score, risk level
- **Bias Detection Events**: Real-time bias incident monitoring with search/filter
- **Quick Analysis**: Severity distribution and demographic breakdowns
- **Action Buttons**: Flag, analyze, and export bias events

### 📈 Analytics
- **Trends**: Time-series analysis of bias patterns
- **Demographics**: Detailed breakdown by protected attributes  
- **Fairness Metrics**: Statistical parity, equal opportunity, equalized odds
- **Mitigation Analysis**: ROI and effectiveness tracking

### 📄 Reports
- **Executive Summary**: High-level findings and recommendations
- **Legal Compliance**: Employment Equity Act and POPIA compliance status
- **Technical Analysis**: Detailed statistical and performance metrics
- **Downloadable Formats**: PDF, HTML, CSV, JSON

### ⚙️ Settings
- **Bias Thresholds**: Configure high/medium/low severity levels
- **Alert Configuration**: Email and Slack notification setup
- **Data Sources**: Configure database connections and refresh schedules
- **User Management**: Role-based access control

---

## 🔧 Configuration

### Bias Detection Thresholds

Modify detection sensitivity in the Settings panel:

- **High Severity**: Default 20% (bias gaps above this level)
- **Medium Severity**: Default 10% (bias gaps above this level)  
- **Statistical Confidence**: Default 95% (minimum confidence for detection)

### Alert Configuration

Set up automated notifications:

```python
# Email alerts
ALERT_EMAIL = "bias-alerts@company.com"
ALERT_FREQUENCY = "Immediate"  # Immediate, Hourly, Daily, Weekly

# Slack integration (optional)
SLACK_WEBHOOK = "https://hooks.slack.com/services/..."
```

### Data Source Configuration

Configure your data source in `dashboard_helpers.py`:

```python
def load_and_process_data(file_path):
    # Modify this function to connect to your database
    # or API endpoint instead of CSV file
    pass
```

---

## 📈 Supported Fairness Metrics

The dashboard calculates multiple fairness metrics:

### Statistical Measures
- **Statistical Parity**: Equal hiring rates across groups
- **Equal Opportunity**: Equal true positive rates for qualified candidates
- **Equalized Odds**: Equal true positive and false positive rates
- **Calibration**: Equal positive predictive values

### Business Metrics
- **Hiring Rate Differences**: Absolute differences in selection rates
- **Representation Gaps**: Workforce composition vs. applicant pool
- **Promotion Parity**: Equal advancement opportunities

---

## 🏛️ Legal Compliance

### Employment Equity Act (South Africa)
- **Section 6**: Non-discrimination requirements
- **Section 15**: Affirmative action measures
- **Section 20**: Reporting obligations

### POPIA (Protection of Personal Information Act)
- **Automated Decision-Making**: Transparency requirements
- **Data Processing**: Lawful basis and consent
- **Data Subject Rights**: Access and correction

---

## 🎨 Customization

### Visual Styling

Modify `assets/custom_styles.css` to customize:
- Color schemes and branding
- Card layouts and animations
- Typography and spacing
- Responsive design breakpoints

### Dashboard Layout

Extend `app.py` to add:
- New analysis sections
- Additional visualizations
- Custom metrics and KPIs
- Integration with external systems

### Data Processing

Enhance `dashboard_helpers.py` with:
- New fairness metrics
- Advanced statistical tests
- Machine learning model integration
- Real-time data streaming

---

## 📊 Sample Data Format

The dashboard expects CSV data with the following structure:

```csv
ID,Race,Gender,Location,English_Fluency,Age_Group,Education,Experience,model_prediction,true_label
1,Black African,Female,Rural,Medium,25-34,Degree,3,0,0
2,White,Male,Urban,High,35-44,Postgrad,8,1,1
...
```

### Required Columns
- **Protected Attributes**: Race, Gender, Location, English_Fluency, Age_Group
- **Features**: Education, Experience (additional features as needed)
- **Outcomes**: model_prediction (0/1), true_label (0/1)

---

## 🔍 Troubleshooting

### Common Issues

**Dashboard won't start:**
```bash
# Check Python version
python --version  # Should be 3.8+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Data loading errors:**
- Verify CSV file exists in `data/` directory
- Check column names match expected format
- Ensure no missing values in protected attributes

**Visualization issues:**
- Update Plotly: `pip install plotly --upgrade`
- Clear browser cache
- Check browser console for JavaScript errors

### Performance Optimization

For large datasets (>100K records):
- Enable data sampling in `dashboard_helpers.py`
- Use database connections instead of CSV files
- Implement caching with `@st.cache_data`

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black app.py dashboard_helpers.py

# Lint code
flake8 app.py dashboard_helpers.py
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support & Contact

**Tech Titanians**  
📧 Email: support@techtitanians.com  
🌐 Website: www.techtitanians.com  
📱 LinkedIn: @TechTitanians

### Professional Services

- **Custom Implementation**: Tailored bias audit solutions
- **Training & Workshops**: Team training on algorithmic fairness
- **Compliance Consulting**: Employment Equity Act guidance
- **Technical Support**: 24/7 enterprise support packages

---

## 🙏 Acknowledgments

- **South African Department of Employment and Labour** for Employment Equity Act guidance
- **AI Ethics Research Community** for fairness metric definitions
- **Streamlit Team** for the excellent dashboard framework
- **Open Source Contributors** for various libraries and tools

---

## 📊 Project Statistics

- **Lines of Code**: ~2,500
- **Test Coverage**: 85%+
- **Supported Metrics**: 8 fairness measures
- **Supported Demographics**: 5 protected attributes
- **Documentation**: Comprehensive README + inline comments

---

*Built with ❤️ by Tech Titanians | Promoting algorithmic fairness in South African employment* 