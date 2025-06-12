# Bias Audit Frontend Dashboard

A modern, interactive web interface for conducting comprehensive ML model bias audits. This dashboard provides an intuitive way to analyze fairness metrics across different demographic groups and visualize bias patterns in machine learning models.

## 🚀 Features

### Core Functionality
- **Interactive File Upload**: Drag-and-drop CSV file upload with real-time validation
- **Dynamic Configuration**: Automatic column detection and configuration setup
- **Comprehensive Analysis**: Full bias audit using multiple fairness metrics
- **Rich Visualizations**: Interactive charts showing bias patterns and distributions
- **Export Capabilities**: Download results as CSV, generate reports, and export charts

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

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.7 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation Steps

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Flask Server**
   ```bash
   python app.py
   ```

3. **Open the Dashboard**
   - Navigate to `http://localhost:5000` in your web browser
   - Or simply open `index.html` directly for client-side only mode

## 📊 How to Use

### Step 1: Upload Your Data
1. **Drag and Drop**: Drag a CSV file onto the upload area
2. **Browse Files**: Click "Choose File" to select a CSV file
3. **Validation**: File is automatically validated and processed

### Step 2: Configure Analysis
1. **Target Column**: Select the column containing true labels
2. **Prediction Column**: Select the column containing model predictions
3. **Protected Attributes**: Check boxes for demographic attributes to analyze (e.g., gender, race, age)

### Step 3: Run Bias Audit
1. Click "Run Bias Audit" to start the analysis
2. View real-time progress with loading indicators
3. Results appear automatically when analysis completes

### Step 4: Analyze Results
1. **Summary Cards**: Overview of bias severity and key metrics
2. **Interactive Charts**: Explore different visualizations
3. **Detailed Table**: Review all fairness metrics by group
4. **Export Options**: Download results in various formats

## 📁 Data Requirements

### File Format
- **Type**: CSV files only
- **Size**: Maximum 16MB
- **Headers**: First row must contain column names

### Data Structure
Your CSV file should contain:
- **Target Column**: Binary labels (0/1) representing true outcomes
- **Prediction Column**: Binary predictions (0/1) from your ML model
- **Protected Attributes**: Categorical columns (e.g., gender, race, education)
- **Sample Size**: Minimum 50 records per protected group recommended

### Example Data Structure
```csv
age,gender,race,education,true_label,predictions
25,Female,White,Bachelor,1,1
45,Male,Black,Master,0,1
30,Female,Hispanic,High School,1,0
...
```

## 🎨 User Interface Guide

### Dashboard Sections

#### 1. Header
- **Title**: Bias Audit Dashboard
- **Subtitle**: Comprehensive ML Model Fairness Analysis

#### 2. Upload Section
- **Upload Area**: Drag-and-drop zone with visual feedback
- **Configuration Panel**: Column selection and protected attribute choices
- **Run Button**: Initiates bias audit with loading animation

#### 3. Results Section
- **Summary Cards**: Key metrics overview
- **Charts Grid**: Four interactive visualizations
- **Results Table**: Detailed fairness metrics
- **Export Panel**: Multiple export options

### Visual Indicators
- **🟢 Low Bias**: Green indicators for acceptable bias levels
- **🟡 Moderate Bias**: Yellow indicators for concerning bias
- **🔴 High Bias**: Red indicators for significant bias
- **🟣 Severe Bias**: Purple indicators for critical bias

## 📈 Understanding Results

### Bias Severity Levels
- **Low (< 5%)**: Acceptable bias level
- **Moderate (5-10%)**: Some bias present, monitor closely
- **High (10-20%)**: Significant bias requiring attention
- **Severe (> 20%)**: Critical bias needing immediate action

### Key Metrics Explained
- **Demographic Parity Difference**: Gap in positive prediction rates
- **Equalized Odds Difference**: Gap in true/false positive rates
- **Accuracy Difference**: Gap in prediction accuracy
- **Sample Size**: Number of records per group

### Chart Interpretations
1. **Accuracy Chart**: Lower bars indicate groups with reduced accuracy
2. **Severity Pie**: Larger red/purple sections indicate more bias
3. **Fairness Radar**: Irregular shapes show metric inconsistencies
4. **Demographic Bar**: Uneven bars show prediction rate disparities

## 🔧 Troubleshooting

### Common Issues

#### File Upload Problems
- **Error**: "Please upload a CSV file"
  - **Solution**: Ensure file has .csv extension
- **Error**: "Error parsing CSV file"
  - **Solution**: Check file format and encoding (UTF-8 recommended)

#### Configuration Issues
- **Error**: "Please select a target column"
  - **Solution**: Choose the column containing true labels (0/1)
- **Error**: "Please select at least one protected attribute"
  - **Solution**: Check at least one demographic column

#### Analysis Errors
- **Error**: "Columns not found in data"
  - **Solution**: Verify selected columns exist in uploaded file
- **Error**: "Error running bias audit"
  - **Solution**: Check data quality and column types

### Data Quality Tips
- Ensure binary columns contain only 0/1 values
- Remove rows with missing values in key columns
- Verify protected attributes have meaningful group sizes
- Check for data encoding issues

## 🚀 Advanced Features

### API Integration
The dashboard includes RESTful API endpoints for programmatic access:
- `POST /api/upload` - File upload
- `POST /api/audit` - Run bias audit
- `POST /api/sample-data` - Generate sample data
- `GET /api/export/{type}/{filename}` - Export results

### Sample Data Generation
Click "Generate Sample Data" to create demonstration data with:
- 1000 synthetic records
- Multiple demographic attributes
- Realistic bias patterns
- Pre-configured column mappings

### Export Options
- **CSV Export**: Download detailed results as spreadsheet
- **Report Generation**: Create formatted text reports
- **Chart Export**: Save visualizations as images

## 🎯 Best Practices

### Data Preparation
1. Clean data before upload (remove missing values)
2. Ensure sufficient sample sizes per group (50+ recommended)
3. Use consistent encoding for categorical variables
4. Validate binary columns contain only 0/1

### Analysis Workflow
1. Start with sample data to understand the interface
2. Upload your actual data and review column detection
3. Select appropriate protected attributes for your use case
4. Run initial audit and review summary metrics
5. Dive deeper into specific charts and tables
6. Export results for documentation and reporting

### Interpretation Guidelines
1. Focus on overall bias severity first
2. Identify which protected attributes show highest bias
3. Look for patterns across multiple fairness metrics
4. Consider practical significance, not just statistical differences
5. Document findings and recommended actions

## 🤝 Support

For technical support or feature requests:
1. Check this documentation first
2. Review troubleshooting section
3. Examine browser console for error messages
4. Ensure all dependencies are properly installed

## 📝 License

This bias audit tool is designed to promote fairness in AI systems. Use responsibly to identify and address bias in machine learning models.

---

**Built for fairness in AI** 🛡️ 