# 🛡️ Bias Audit Dashboard

A comprehensive web-based tool for analyzing bias in machine learning models. This dashboard provides an intuitive interface to upload data, configure fairness metrics, and visualize bias patterns across different demographic groups.

![Bias Audit Dashboard](https://img.shields.io/badge/Status-Active-green) ![Python](https://img.shields.io/badge/Python-3.7+-blue) ![Flask](https://img.shields.io/badge/Flask-2.0+-red) ![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow)

## 🚀 Features

### Core Functionality
- **📊 Interactive File Upload**: Drag-and-drop CSV file upload with real-time validation
- **⚙️ Dynamic Configuration**: Automatic column detection and configuration setup
- **🔍 Comprehensive Analysis**: Full bias audit using multiple fairness metrics
- **📈 Rich Visualizations**: Interactive charts showing bias patterns and distributions
- **💾 Export Capabilities**: Download results as CSV, generate reports, and export charts

### Bias Metrics Analyzed
- **Demographic Parity**: Equal positive prediction rates across groups
- **Equalized Odds**: Equal true positive and false positive rates
- **Accuracy Fairness**: Consistent accuracy across protected groups
- **Precision/Recall**: Performance consistency analysis
- **F1 Score**: Balanced performance metrics

### Visualization Types
- **Accuracy Comparison**: Bar charts showing accuracy by protected groups
- **Bias Severity Distribution**: Pie charts of bias severity levels
- **Fairness Metrics Radar**: Multi-dimensional fairness comparison
- **Demographic Parity**: Positive prediction rate analysis

## 🛠️ Quick Start

### Prerequisites
- Python 3.7 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mooncakeSG/bias-audit-report-.git
   cd bias-audit-report-
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate sample data (optional)**
   ```bash
   python generate_sample_data.py
   ```

4. **Start the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:5000`

## 📊 Usage Guide

### Step 1: Upload Data
- **Option A**: Drag and drop a CSV file onto the upload area
- **Option B**: Click "Choose File" to browse and select a CSV
- **Option C**: Click "📊 Use Sample Data" for a quick demo

### Step 2: Configure Analysis
1. **Target Column**: Select the column containing true labels (0/1)
2. **Prediction Column**: Select the column containing model predictions (0/1)
3. **Protected Attributes**: Check the demographic attributes to analyze

### Step 3: Run Analysis
- Click "Run Bias Audit" to start the analysis
- View real-time progress indicators
- Explore interactive results automatically displayed

### Step 4: Interpret Results
- **Summary Cards**: Key bias metrics overview
- **Interactive Charts**: Multiple visualization perspectives
- **Detailed Table**: Comprehensive fairness metrics by group
- **Export Options**: Download results for further analysis

## 📁 File Structure

```
bias-audit-report-/
├── 📄 index.html              # Main dashboard interface
├── 🎨 styles.css              # Modern CSS styling
├── ⚡ script.js               # Interactive JavaScript functionality
├── 🐍 app.py                  # Flask backend server
├── 🔧 bias_audit.py           # Core bias analysis engine
├── 📊 generate_sample_data.py # Sample data generator
├── 📋 requirements.txt        # Python dependencies
├── 📈 sample_bias_data.csv    # Example dataset with bias patterns
├── 📚 README_frontend.md      # Detailed frontend documentation
└── 📖 README.md               # This file
```

## 🎯 Data Requirements

### File Format
- **Type**: CSV files only
- **Size**: Maximum 16MB
- **Headers**: First row must contain column names
- **Encoding**: UTF-8 recommended

### Required Columns
- **Target Column**: Binary labels (0/1) representing true outcomes
- **Prediction Column**: Binary predictions (0/1) from your ML model
- **Protected Attributes**: Categorical columns (e.g., gender, race, age_group)

### Example Data Structure
```csv
age,gender,race,education,true_label,model_prediction
25,Female,White,Bachelor,1,1
45,Male,Black,Master,0,1
30,Female,Hispanic,High School,1,0
```

## 📈 Understanding Results

### Bias Severity Levels
- **🟢 Low (< 5%)**: Acceptable bias level
- **🟡 Moderate (5-10%)**: Some bias present, monitor closely
- **🔴 High (10-20%)**: Significant bias requiring attention
- **🟣 Severe (> 20%)**: Critical bias needing immediate action

### Key Metrics
- **Demographic Parity Difference**: Gap in positive prediction rates between groups
- **Equalized Odds Difference**: Gap in true/false positive rates between groups
- **Accuracy Difference**: Gap in prediction accuracy between groups

## 🌐 API Endpoints

The Flask backend provides RESTful API endpoints:

- `GET /` - Serve main dashboard
- `POST /api/upload` - Handle file uploads
- `POST /api/audit` - Run bias audit analysis
- `POST /api/sample-data` - Generate sample data
- `GET /api/export/{type}/{filename}` - Export results

## 🧪 Sample Data

The repository includes a sample dataset (`sample_bias_data.csv`) with intentional bias patterns:

- **500 records** with realistic demographic data
- **Gender bias**: Males favored in predictions (+15% boost)
- **Racial bias**: White and Asian groups favored (+12% and +8% respectively)
- **Education correlation**: Higher education leads to better predictions

Perfect for testing and understanding bias detection capabilities!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built with Flask, Chart.js, and modern web technologies
- Inspired by the need for fairness and transparency in AI systems
- Designed to promote responsible AI development practices

## 📞 Support

- **Documentation**: See `README_frontend.md` for detailed usage instructions
- **Issues**: Report bugs or request features via GitHub Issues
- **Questions**: Check existing issues or create a new one

---

**🛡️ Built for fairness in AI - Help make machine learning more equitable!** 