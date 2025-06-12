#!/usr/bin/env python3
"""
Flask Web Application for Bias Audit Tool
==========================================

This Flask app provides a web interface and API endpoints for the bias audit tool.
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import tempfile
from datetime import datetime
import traceback

# Import the bias audit functionality
from bias_audit import BiasAuditor, generate_sample_data

app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

@app.route('/')
def index():
    """Serve the main dashboard page."""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, etc.)."""
    return send_from_directory('.', filename)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and return column information."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'}), 400
        
        # Save the uploaded file
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and analyze the CSV
        try:
            df = pd.read_csv(filepath)
            
            # Get column information
            columns = df.columns.tolist()
            data_types = df.dtypes.to_dict()
            sample_data = df.head(5).to_dict('records')
            
            # Identify potential columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            return jsonify({
                'success': True,
                'filename': filename,
                'columns': columns,
                'data_types': {col: str(dtype) for col, dtype in data_types.items()},
                'sample_data': sample_data,
                'numeric_columns': numeric_columns,
                'categorical_columns': categorical_columns,
                'row_count': len(df),
                'column_count': len(columns)
            })
            
        except Exception as e:
            return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/audit', methods=['POST'])
def run_bias_audit():
    """Run bias audit on uploaded data."""
    try:
        data = request.get_json()
        
        # Validate required parameters
        required_fields = ['filename', 'target_column', 'prediction_column', 'protected_attributes']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        filename = data['filename']
        target_column = data['target_column']
        prediction_column = data['prediction_column']
        protected_attributes = data['protected_attributes']
        
        if not protected_attributes:
            return jsonify({'error': 'At least one protected attribute must be selected'}), 400
        
        # Load the data file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Data file not found'}), 404
        
        df = pd.read_csv(filepath)
        
        # Validate columns exist
        all_columns = [target_column, prediction_column] + protected_attributes
        missing_columns = [col for col in all_columns if col not in df.columns]
        if missing_columns:
            return jsonify({'error': f'Columns not found in data: {missing_columns}'}), 400
        
        # Initialize bias auditor
        auditor = BiasAuditor(
            data=df,
            protected_attributes=protected_attributes,
            target_column=target_column,
            prediction_column=prediction_column
        )
        
        # Run the audit
        fairness_results = auditor.calculate_fairness_metrics()
        bias_metrics = auditor.calculate_bias_metrics()
        
        # Generate visualization
        chart_path = f"bias_audit_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        auditor.generate_visualizations(save_path=chart_path)
        
        # Save results
        results_csv = f"bias_audit_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_csv = f"bias_audit_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        fairness_results.to_csv(results_csv, index=False)
        bias_metrics.to_csv(summary_csv, index=False)
        
        # Prepare response data
        response_data = {
            'success': True,
            'fairness_results': fairness_results.to_dict('records'),
            'bias_metrics': bias_metrics.to_dict('records'),
            'chart_path': chart_path,
            'results_csv': results_csv,
            'summary_csv': summary_csv,
            'summary': {
                'total_samples': int(fairness_results['sample_size'].sum()),
                'attributes_analyzed': len(protected_attributes),
                'overall_severity': get_overall_severity(bias_metrics),
                'max_bias_score': float(get_max_bias_score(bias_metrics))
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in bias audit: {traceback.format_exc()}")
        return jsonify({'error': f'Audit failed: {str(e)}'}), 500

@app.route('/api/sample-data', methods=['POST'])
def generate_sample():
    """Generate sample data for demonstration."""
    try:
        # Generate sample data
        sample_df = generate_sample_data()
        
        # Save sample data
        filename = f"sample_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        sample_df.to_csv(filepath, index=False)
        
        # Get column information
        columns = sample_df.columns.tolist()
        data_types = sample_df.dtypes.to_dict()
        sample_data = sample_df.head(5).to_dict('records')
        
        return jsonify({
            'success': True,
            'filename': filename,
            'columns': columns,
            'data_types': {col: str(dtype) for col, dtype in data_types.items()},
            'sample_data': sample_data,
            'numeric_columns': ['age'],
            'categorical_columns': ['gender', 'race', 'education'],
            'row_count': len(sample_df),
            'column_count': len(columns),
            'suggested_config': {
                'target_column': 'true_label',
                'prediction_column': 'predictions',
                'protected_attributes': ['gender', 'race', 'education']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate sample data: {str(e)}'}), 500

@app.route('/api/export/<export_type>/<filename>')
def export_file(export_type, filename):
    """Export files (CSV, charts, reports)."""
    try:
        if export_type == 'csv':
            return send_from_directory('.', filename, as_attachment=True)
        elif export_type == 'chart':
            return send_from_directory('.', filename, as_attachment=True)
        else:
            return jsonify({'error': 'Invalid export type'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

def get_overall_severity(bias_metrics_df):
    """Determine overall bias severity."""
    if bias_metrics_df.empty:
        return 'N/A'
    
    severities = bias_metrics_df['bias_severity'].tolist()
    
    if 'Severe' in severities:
        return 'Severe'
    elif 'High' in severities:
        return 'High'
    elif 'Moderate' in severities:
        return 'Moderate'
    else:
        return 'Low'

def get_max_bias_score(bias_metrics_df):
    """Get maximum bias score across all metrics."""
    if bias_metrics_df.empty:
        return 0.0
    
    max_scores = []
    for col in ['demographic_parity_difference', 'equalized_odds_difference', 'accuracy_difference']:
        if col in bias_metrics_df.columns:
            max_scores.append(bias_metrics_df[col].max())
    
    return max(max_scores) if max_scores else 0.0

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    return jsonify({'error': 'Internal server error occurred.'}), 500

if __name__ == '__main__':
    print("Starting Bias Audit Dashboard...")
    print("Access the dashboard at: http://localhost:5000")
    print("\nFeatures:")
    print("- Upload CSV files for bias analysis")
    print("- Configure protected attributes and target columns")
    print("- Generate comprehensive bias audit reports")
    print("- Interactive visualizations and charts")
    print("- Export results in multiple formats")
    print("\nPress Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 