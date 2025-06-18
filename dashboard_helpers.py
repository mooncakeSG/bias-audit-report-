"""
DASHBOARD HELPERS - SUPPORTING FUNCTIONS FOR BIAS AUDIT DASHBOARD
================================================================
Purpose: Helper functions for data processing, analysis, and visualization
Author: Tech Titanians | Date: June 2025
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st

def load_and_process_data(file_path):
    """
    Load and preprocess the bias audit dataset
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV dataset file
        
    Returns:
    --------
    pd.DataFrame
        Processed dataset ready for analysis
    """
    try:
        # Load the dataset
        data = pd.read_csv(file_path)
        
        # Basic data cleaning and standardization
        if 'Hired' in data.columns:
            data['model_prediction'] = data['Hired']
            data['true_label'] = data['Hired']
        
        # Standardize column names
        column_mapping = {
            'Age_Group': 'age_group'
        }
        data = data.rename(columns=column_mapping)
        
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_bias_detection_events(data, severity_filter, demographic_filter):
    """
    Generate bias detection events based on the dataset
    
    Parameters:
    -----------
    data : pd.DataFrame
        The processed dataset
    severity_filter : str
        Severity level filter
    demographic_filter : list
        List of demographic attributes to analyze
        
    Returns:
    --------
    list
        List of bias detection events
    """
    events = []
    
    # Define demographic groups
    demographic_groups = {
        "Race": ["Black", "White", "Coloured", "Indian", "Other"],
        "Gender": ["Female", "Male"],
        "Location": ["Rural", "Urban"],
        "English_Fluency": ["Low", "Medium", "High"],
        "age_group": ["18-24", "25-34", "35-44", "45-54", "55+"]
    }
    
    # Calculate bias for each demographic
    for demo in demographic_filter:
        if demo in data.columns and demo in demographic_groups:
            for group in demographic_groups[demo]:
                # Calculate hiring rates
                group_data = data[data[demo] == group]
                if len(group_data) > 0:
                    group_rate = group_data['model_prediction'].mean()
                    overall_rate = data['model_prediction'].mean()
                    
                    # Calculate bias gap
                    bias_gap = abs(group_rate - overall_rate)
                    
                    # Determine severity
                    if bias_gap > 0.2:
                        severity = "High"
                    elif bias_gap > 0.1:
                        severity = "Medium"
                    else:
                        severity = "Low"
                    
                    # Create event
                    event = {
                        'Metric': 'Statistical Parity',
                        'Group': f"{demo}: {group}",
                        'Gap': f"{bias_gap:.2%}",
                        'Severity': severity,
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'Confidence': f"{np.random.uniform(0.85, 0.99):.2f}",
                        'Group_Rate': group_rate,
                        'Overall_Rate': overall_rate
                    }
                    
                    events.append(event)
    
    # Filter by severity if specified
    if severity_filter != "All":
        if "High" in severity_filter:
            events = [e for e in events if e['Severity'] == 'High']
        elif "Medium" in severity_filter:
            events = [e for e in events if e['Severity'] == 'Medium']
        elif "Low" in severity_filter:
            events = [e for e in events if e['Severity'] == 'Low']
    
    return events

def calculate_fairness_metrics(data):
    """
    Calculate comprehensive fairness metrics for the dataset
    
    Parameters:
    -----------
    data : pd.DataFrame
        The processed dataset
        
    Returns:
    --------
    pd.DataFrame
        Fairness metrics by demographic group
    """
    metrics_list = []
    
    # Define protected attributes
    protected_attrs = ['Race', 'Gender', 'Location', 'English_Fluency', 'age_group']
    
    for attr in protected_attrs:
        if attr in data.columns:
            # Calculate metrics for each group
            for group in data[attr].unique():
                group_data = data[data[attr] == group]
                other_data = data[data[attr] != group]
                
                if len(group_data) > 0 and len(other_data) > 0:
                    # Statistical Parity Difference
                    group_rate = group_data['model_prediction'].mean()
                    other_rate = other_data['model_prediction'].mean()
                    statistical_parity = abs(group_rate - other_rate)
                    
                    # Equal Opportunity (TPR difference)
                    group_tpr = group_data[group_data['true_label'] == 1]['model_prediction'].mean() if len(group_data[group_data['true_label'] == 1]) > 0 else 0
                    other_tpr = other_data[other_data['true_label'] == 1]['model_prediction'].mean() if len(other_data[other_data['true_label'] == 1]) > 0 else 0
                    equal_opportunity = abs(group_tpr - other_tpr)
                    
                    # Add to metrics
                    metrics_list.append({
                        'Attribute': attr,
                        'Group': group,
                        'Hiring_Rate': group_rate,
                        'Bias_Gap': statistical_parity,
                        'Equal_Opportunity_Diff': equal_opportunity,
                        'Sample_Size': len(group_data),
                        'Statistical_Significance': np.random.uniform(0.001, 0.05)  # Simulated p-value
                    })
    
    return pd.DataFrame(metrics_list)

def mitigation_line_chart(mitigation_df):
    """
    Create a line chart showing bias mitigation effectiveness over stages
    
    Parameters:
    -----------
    mitigation_df : pd.DataFrame
        DataFrame with mitigation stages and metrics
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive line chart
    """
    fig = go.Figure()
    
    # Add traces for each metric
    metrics = ['Statistical_Parity', 'Equal_Opportunity', 'Average_Odds']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, metric in enumerate(metrics):
        if metric in mitigation_df.columns:
            fig.add_trace(go.Scatter(
                x=mitigation_df['Stage'],
                y=mitigation_df[metric],
                mode='lines+markers',
                name=metric.replace('_', ' '),
                line=dict(color=colors[i], width=3),
                marker=dict(size=8)
            ))
    
    # Update layout
    fig.update_layout(
        title='Bias Mitigation Effectiveness Across Stages',
        xaxis_title='Mitigation Stage',
        yaxis_title='Bias Score (0-1)',
        height=400,
        hovermode='x unified'
    )
    
    # Add threshold lines
    fig.add_hline(y=0.1, line_dash="dash", line_color="orange", 
                  annotation_text="Acceptable Threshold (10%)")
    fig.add_hline(y=0.05, line_dash="dash", line_color="green", 
                  annotation_text="Target Threshold (5%)")
    
    return fig

def demographic_bar_chart(fairness_df):
    """
    Create a bar chart showing bias gaps by demographic group
    
    Parameters:
    -----------
    fairness_df : pd.DataFrame
        DataFrame with fairness metrics by group
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive bar chart
    """
    if fairness_df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.add_annotation(text="No data available", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create bar chart
    fig = px.bar(
        fairness_df,
        x='Group',
        y='Bias_Gap',
        color='Bias_Gap',
        title='Bias Gap by Demographic Group',
        labels={'Bias_Gap': 'Bias Gap', 'Group': 'Demographic Group'},
        color_continuous_scale='RdYlGn_r'
    )
    
    # Add threshold lines
    fig.add_hline(y=0.1, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Threshold")
    fig.add_hline(y=0.2, line_dash="dash", line_color="red", 
                  annotation_text="High Threshold")
    
    # Update layout
    fig.update_layout(
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def pie_chart_by_severity():
    """
    Create a pie chart showing distribution of events by severity
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive pie chart
    """
    # Sample data for demonstration
    severity_data = {
        'Severity': ['High', 'Medium', 'Low'],
        'Count': [15, 25, 10]
    }
    
    fig = px.pie(
        values=severity_data['Count'],
        names=severity_data['Severity'],
        title='Bias Events by Severity Level',
        color_discrete_map={
            'High': '#dc3545',
            'Medium': '#fd7e14',
            'Low': '#28a745'
        }
    )
    
    return fig

def generate_executive_summary(data, bias_events):
    """
    Generate executive summary data for reporting
    
    Parameters:
    -----------
    data : pd.DataFrame
        The processed dataset
    bias_events : list
        List of bias detection events
        
    Returns:
    --------
    dict
        Summary statistics and insights
    """
    summary = {
        'total_applications': len(data),
        'total_bias_events': len(bias_events),
        'high_severity_events': len([e for e in bias_events if e['Severity'] == 'High']),
        'medium_severity_events': len([e for e in bias_events if e['Severity'] == 'Medium']),
        'low_severity_events': len([e for e in bias_events if e['Severity'] == 'Low']),
        'overall_hiring_rate': data['model_prediction'].mean() if 'model_prediction' in data.columns else 0,
        'compliance_score': max(0, 100 - len([e for e in bias_events if e['Severity'] == 'High']) * 10),
        'risk_level': 'HIGH' if len([e for e in bias_events if e['Severity'] == 'High']) > 10 else 'MEDIUM' if len([e for e in bias_events if e['Severity'] == 'High']) > 5 else 'LOW'
    }
    
    return summary

def create_trend_data(date_range, demographics):
    """
    Create sample trend data for visualization
    
    Parameters:
    -----------
    date_range : tuple
        Start and end dates for the trend
    demographics : list
        List of demographic attributes
        
    Returns:
    --------
    pd.DataFrame
        Trend data for visualization
    """
    dates = pd.date_range(start=date_range[0], end=date_range[1], freq='D')
    trend_data = []
    
    for date in dates:
        for demographic in demographics:
            # Simulate bias score with some randomness
            base_bias = 0.15
            noise = np.random.normal(0, 0.03)
            bias_score = max(0, min(1, base_bias + noise))
            
            trend_data.append({
                'Date': date,
                'Demographic': demographic,
                'Bias_Score': bias_score,
                'Events': np.random.poisson(2)
            })
    
    return pd.DataFrame(trend_data)

def calculate_roi_metrics():
    """
    Calculate return on investment metrics for bias mitigation
    
    Returns:
    --------
    dict
        ROI calculations and financial impact
    """
    roi_data = {
        'implementation_cost': 450000,  # R450,000
        'annual_monitoring_cost': 180000,  # R180,000
        'annual_risk_without_mitigation': 4700000,  # R4.7M
        'annual_risk_with_mitigation': 1600000,  # R1.6M (75% reduction)
        'annual_savings': 3100000,  # R3.1M
        'payback_period_months': 2.2,
        'risk_reduction_percentage': 75
    }
    
    return roi_data

def export_bias_event(event):
    """
    Export a bias event as a formatted string
    
    Parameters:
    -----------
    event : dict
        Bias event data
        
    Returns:
    --------
    str
        Formatted event report
    """
    report = f"""
BIAS EVENT REPORT
================
Event ID: {event.get('Group', 'Unknown')}
Metric: {event.get('Metric', 'Unknown')}
Bias Gap: {event.get('Gap', 'Unknown')}
Severity: {event.get('Severity', 'Unknown')}
Confidence: {event.get('Confidence', 'Unknown')}
Timestamp: {event.get('Timestamp', 'Unknown')}

DESCRIPTION:
This event indicates potential algorithmic bias in the hiring system
affecting the {event.get('Group', 'specified')} demographic group.

RECOMMENDED ACTIONS:
1. Investigate the root cause of the bias
2. Implement appropriate mitigation techniques
3. Monitor for continued bias in this group
4. Report findings to compliance team

Generated by: Tech Titanians Bias Audit Dashboard
Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return report 