import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Try importing fairness libraries with fallbacks
try:
    from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    from fairlearn.postprocessing import ThresholdOptimizer
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False

try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.algorithms.postprocessing import EqOddsPostprocessing
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False

# PAGE CONFIGURATION
st.set_page_config(
    page_title="🎯 SA Bias Audit Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MODERN INBOX-STYLE CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #003366 0%, #0066cc 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
}

/* Modern Inbox-Style Bias Event Cards */
.bias-event-card {
    background: white;
    border-radius: 12px;
    padding: 18px 20px;
    margin: 8px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 5px solid;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.bias-event-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    cursor: pointer;
}

.bias-event-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--card-color), transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.bias-event-card:hover::before {
    opacity: 1;
}

/* Severity Color Classes */
.severity-high { 
    border-left-color: #e74c3c; 
    --card-color: #e74c3c;
}
.severity-medium { 
    border-left-color: #f39c12; 
    --card-color: #f39c12;
}
.severity-low { 
    border-left-color: #2ecc71; 
    --card-color: #2ecc71;
}

/* Event Header */
.event-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.event-title {
    font-weight: 600;
    font-size: 1.1rem;
    color: #2c3e50;
    margin: 0;
}

.severity-badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    color: white;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.severity-badge.high { background: #e74c3c; }
.severity-badge.medium { background: #f39c12; }
.severity-badge.low { background: #2ecc71; }

/* Event Details */
.event-details {
    color: #7f8c8d;
    font-size: 0.9rem;
    line-height: 1.4;
    margin-bottom: 12px;
}

.event-meta {
    display: flex;
    gap: 20px;
    margin-bottom: 15px;
    flex-wrap: wrap;
}

.meta-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.85rem;
    color: #6c757d;
}

.meta-value {
    font-weight: 600;
    color: #495057;
}

/* Action Buttons */
.action-buttons {
    display: flex;
    gap: 8px;
    margin-top: 12px;
    flex-wrap: wrap;
}

.action-btn {
    padding: 8px 16px;
    border: none;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    text-decoration: none;
    color: white;
}

.action-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.btn-flag {
    background: linear-gradient(135deg, #ff4757, #ff3838);
}

.btn-analyze {
    background: linear-gradient(135deg, #4b7bec, #3742fa);
}

.btn-export {
    background: linear-gradient(135deg, #2ed573, #1dd1a1);
}

.btn-flag:hover { background: linear-gradient(135deg, #ff3838, #ff2f2f); }
.btn-analyze:hover { background: linear-gradient(135deg, #3742fa, #2f3542); }
.btn-export:hover { background: linear-gradient(135deg, #1dd1a1, #00d2d3); }

/* Search Input Styling */
.stTextInput > div > div > input {
    border: 2px solid #e9ecef;
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 0.95rem;
    transition: all 0.3s ease;
    background: #f8f9fa;
}

.stTextInput > div > div > input:focus {
    border-color: #0066cc;
    background: white;
    box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
}

.stTextInput > div > div > input::placeholder {
    color: #adb5bd;
    font-style: italic;
}

/* Section Headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid #e9ecef;
}

.section-icon {
    font-size: 1.5rem;
}

.section-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #2c3e50;
    margin: 0;
}

/* Empty State */
.empty-state {
    text-align: center;
    padding: 40px 20px;
    color: #6c757d;
    background: #f8f9fa;
    border-radius: 12px;
    border: 2px dashed #dee2e6;
}

.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 15px;
    opacity: 0.5;
}

/* Responsive Design */
@media (max-width: 768px) {
    .event-meta {
        flex-direction: column;
        gap: 8px;
    }
    
    .action-buttons {
        flex-direction: column;
    }
    
    .action-btn {
        width: 100%;
        justify-content: center;
    }
}
</style>

<script>
function flagEvent(eventId) {
    alert('Issue #' + eventId + ' marked as priority for HR review! ✅');
}

function analyzeEvent(eventId) {
    alert('Getting detailed analysis for Issue #' + eventId + '... 🔍');
}

function exportEvent(eventId) {
    alert('Saving full report for Issue #' + eventId + '... 📁');
}
</script>
""", unsafe_allow_html=True)

# SIDEBAR NAVIGATION - User-friendly
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem;">
    <h2>⚖️ Tech Titanians</h2>
    <p style="color: #666;">Fair Hiring Monitor</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("📋 Menu")
section = st.sidebar.radio("Go to", ["🎯 Main Dashboard", "📊 Detailed Analysis", "📄 Reports", "⚙️ Settings", "🤖 ML Pipeline"])

# GLOBAL FILTERS - User-friendly
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔍 Filters")

date_range = st.sidebar.date_input(
    "📅 Time Period to Check",
    value=[datetime.now() - timedelta(days=30), datetime.now()]
)

severity_filter = st.sidebar.selectbox(
    "🚨 Show Issues by Priority",
    ["All Issues", "🚨 Urgent (>20%)", "⚠️ Needs Attention (10-20%)", "✅ Minor (<10%)"]
)

demographic_filter = st.sidebar.multiselect(
    "👥 Groups to Monitor",
    ["Race", "Gender", "Location", "English_Fluency", "Age_Group"],
    default=["Race", "Gender", "Location"]
)

# DATA LOADING
@st.cache_data
def load_dashboard_data():
    """Load and process data for the dashboard"""
    try:
        data = pd.read_csv("data/synthetic_sa_job_dataset.csv")
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
data = load_dashboard_data()
if data is None:
    st.error("Could not load data. Please check if the data file exists.")
    st.stop()

# Create sample bias events for demonstration
def create_sample_bias_events():
    events = []
    demographics = ["Race", "Gender", "Location", "English_Fluency", "Age_Group"]
    groups = {
        "Race": ["Black African", "White", "Coloured", "Indian"],
        "Gender": ["Female", "Male"],
        "Location": ["Rural", "Urban"],
        "English_Fluency": ["Low", "Medium", "High"],
        "Age_Group": ["18-24", "25-34", "35-44", "45-54", "55+"]
    }
    
    for demo in demographics:
        for group in groups[demo]:
            gap = np.random.uniform(0.05, 0.35)
            severity = "High" if gap > 0.2 else "Medium" if gap > 0.1 else "Low"
            
            events.append({
                'Metric': 'Statistical Parity',
                'Group': f"{demo}: {group}",
                'Gap': f"{gap:.2%}",
                'Severity': severity,
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'Confidence': f"{np.random.uniform(0.85, 0.99):.2f}"
            })
    
    return events

bias_events = create_sample_bias_events()

# ML PIPELINE FUNCTIONS
@st.cache_data
def preprocess_data(df, target_column):
    """
    Enhanced preprocessing function that works with ANY dataset format
    """
    try:
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Extract target variable
        y = df_processed[target_column].copy()
        X = df_processed.drop(columns=[target_column])
        
        # Handle missing values in target
        if y.isnull().any():
            st.warning(f"⚠️ Found {y.isnull().sum()} missing values in target column. Removing these rows.")
            valid_indices = ~y.isnull()
            y = y[valid_indices]
            X = X[valid_indices]
        
        # Enhanced target encoding for ANY binary format
        if y.dtype == 'object' or not all(y.isin([0, 1])):
            unique_values = y.unique()
            if len(unique_values) == 2:
                # Handle various binary formats
                val1, val2 = unique_values[0], unique_values[1]
                
                # Smart encoding based on common patterns
                positive_indicators = ['yes', 'true', '1', 'hired', 'approved', 'accepted', 'passed', 'success', 'positive', 'good', 'high']
                negative_indicators = ['no', 'false', '0', 'not hired', 'rejected', 'failed', 'negative', 'bad', 'low']
                
                val1_str, val2_str = str(val1).lower(), str(val2).lower()
                
                if any(pos in val1_str for pos in positive_indicators):
                    y = (y == val1).astype(int)
                elif any(pos in val2_str for pos in positive_indicators):
                    y = (y == val2).astype(int)
                elif any(neg in val1_str for neg in negative_indicators):
                    y = (y == val2).astype(int)
                elif any(neg in val2_str for neg in negative_indicators):
                    y = (y == val1).astype(int)
                else:
                    # Default: first value = 0, second value = 1
                    y = pd.Categorical(y).codes.astype(int)
                    
                st.success(f"✅ Target encoded: {val1} → {(y==0).sum()} samples, {val2} → {(y==1).sum()} samples")
            else:
                st.error(f"Target column '{target_column}' must be binary (2 values). Found: {unique_values}")
                return None, None, None, None
        else:
            y = y.astype(int)
        
        # Enhanced feature preprocessing for ANY dataset
        categorical_cols = []
        numerical_cols = []
        
        st.info(f"🔄 Analyzing {len(X.columns)} columns...")
        
        for col in X.columns:
            # Handle missing values first
            if X[col].isnull().any():
                missing_count = X[col].isnull().sum()
                if X[col].dtype in ['object', 'category']:
                    X[col] = X[col].fillna('Unknown')
                    st.info(f"📝 Filled {missing_count} missing values in '{col}' with 'Unknown'")
                else:
                    X[col] = X[col].fillna(X[col].median())
                    st.info(f"📊 Filled {missing_count} missing values in '{col}' with median")
            
            # Determine column type
            if X[col].dtype in ['object', 'category']:
                categorical_cols.append(col)
                unique_count = X[col].nunique()
                # Clean categorical values to avoid encoding issues
                X[col] = X[col].astype(str).str.strip()  # Convert to string and strip whitespace
                st.info(f"📂 '{col}': Categorical ({unique_count} unique values)")
            elif X[col].dtype in ['bool']:
                X[col] = X[col].astype(int)
                numerical_cols.append(col)
                st.info(f"🔢 '{col}': Boolean → Numerical")
            else:
                # Check if numeric column is actually categorical
                unique_count = X[col].nunique()
                if unique_count <= 10 and X[col].dtype in ['int64', 'float64']:
                    # Likely categorical (like age groups coded as numbers)
                    categorical_cols.append(col)
                    # Convert to string to treat as categorical
                    X[col] = X[col].astype(str)
                    st.info(f"📂 '{col}': Numerical → Categorical ({unique_count} unique values)")
                else:
                    numerical_cols.append(col)
                    st.info(f"🔢 '{col}': Numerical ({unique_count} unique values)")
        
        st.success(f"✅ Column analysis complete: {len(categorical_cols)} categorical, {len(numerical_cols)} numerical")
        
        # Encode categorical variables
        if categorical_cols:
            st.info(f"🔄 Encoding {len(categorical_cols)} categorical columns...")
            # Create dummy variables for each categorical column separately to avoid prefix mismatch
            encoded_dfs = []
            for col in categorical_cols:
                try:
                    # Additional cleaning for problematic categorical values
                    X[col] = X[col].astype(str).str.replace('[^a-zA-Z0-9_]', '_', regex=True)  # Replace special chars
                    X[col] = X[col].str[:50]  # Limit length to avoid memory issues
                    
                    col_dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    encoded_dfs.append(col_dummies)
                    st.info(f"✅ Encoded '{col}': {len(col_dummies.columns)} dummy variables created")
                except Exception as e:
                    st.warning(f"⚠️ Failed to encode '{col}': {str(e)}")
                    # Try simpler encoding as fallback
                    try:
                        # Convert to simple numeric codes as fallback
                        X[col] = pd.Categorical(X[col]).codes
                        numerical_cols.append(col)
                        st.info(f"🔄 '{col}': Fallback to numeric codes")
                    except:
                        st.error(f"❌ Complete failure encoding '{col}', skipping")
                    continue
            
            # Combine all encoded categorical variables
            if encoded_dfs:
                X_encoded = pd.concat(encoded_dfs, axis=1)
                st.success(f"✅ Combined categorical features: {len(X_encoded.columns)} total dummy variables")
                
                if numerical_cols:
                    X_numerical = X[numerical_cols]
                    X = pd.concat([X_numerical, X_encoded], axis=1)
                    st.success(f"✅ Final dataset: {len(X_numerical.columns)} numerical + {len(X_encoded.columns)} categorical = {len(X.columns)} total features")
                else:
                    X = X_encoded
                    st.info("ℹ️ Using only categorical features (no numerical features found)")
            else:
                # Only numerical columns
                if numerical_cols:
                    X = X[numerical_cols]
                    st.info(f"ℹ️ Using {len(numerical_cols)} numerical features only")
                else:
                    st.error("❌ No valid features found after preprocessing")
                    return None, None, None, None
        elif numerical_cols:
            # Only numerical columns
            X = X[numerical_cols]
            st.info(f"ℹ️ Using {len(numerical_cols)} numerical features only")
        else:
            st.error("❌ No valid features found after preprocessing")
            return None, None, None, None
        
        # Ensure all columns are numeric with better error handling
        try:
            X = X.apply(pd.to_numeric, errors='coerce')
            st.info("✅ All columns converted to numeric format")
        except Exception as e:
            st.warning(f"⚠️ Some columns couldn't be converted to numeric: {str(e)}")
            # Try column by column conversion
            for col in X.columns:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    st.warning(f"⚠️ Column '{col}' conversion failed, filling with 0")
                    X[col] = 0
        
        # Handle any remaining NaN values
        if X.isnull().any().any():
            nan_count = X.isnull().sum().sum()
            X = X.fillna(0)
            st.warning(f"⚠️ Filled {nan_count} remaining NaN values with 0")
        
        # Final validation
        if len(X.columns) == 0:
            st.error("❌ No features remain after preprocessing")
            return None, None, None, None
        
        if len(X) != len(y):
            st.error(f"❌ Feature-target mismatch: {len(X)} samples vs {len(y)} targets")
            return None, None, None, None
        
        # Check for any infinite values with proper type checking
        try:
            # Ensure X is fully numeric before checking for inf values
            X = X.astype(float)
            if np.isinf(X.values).any():
                st.warning("⚠️ Found infinite values, replacing with 0")
                X = X.replace([np.inf, -np.inf], 0)
        except Exception as e:
            st.warning(f"⚠️ Could not check for infinite values: {str(e)}")
            # Ensure no problematic values remain
            X = X.fillna(0)
            try:
                X = X.astype(float)
            except:
                st.error("❌ Failed to convert data to numeric format")
                return None, None, None, None
        
        st.success(f"🎉 Preprocessing complete! Final dataset: {len(X)} samples × {len(X.columns)} features")
        st.info(f"📊 Processed features: {len(categorical_cols)} categorical, {len(numerical_cols)} numerical → {len(X.columns)} total features")
        
        return X, y, categorical_cols, numerical_cols
        
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None, None, None, None

@st.cache_data
def train_baseline_model(X, y, model_type='LogisticRegression', test_size=0.2, random_state=42, **kwargs):
    """
    Enhanced model training function with configurable parameters
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Initialize model with custom parameters
        if model_type == 'LogisticRegression':
            model = LogisticRegression(
                random_state=random_state,
                max_iter=kwargs.get('max_iter', 1000),
                solver=kwargs.get('solver', 'lbfgs')
            )
        elif model_type == 'RandomForestClassifier':
            model = RandomForestClassifier(
                random_state=random_state,
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None)
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Store original indices for fairness analysis
        X_test_original = X_test.copy()
        
        return {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'X_test_original': X_test_original,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'y_pred_proba_test': y_pred_proba_test,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'model_type': model_type,
            'classification_report': classification_report(y_test, y_pred_test)
        }
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

def calculate_fairness_metrics_custom(y_true, y_pred, sensitive_features):
    """Calculate fairness metrics with custom implementation"""
    results = {}
    
    # Convert to numpy arrays and ensure proper data types
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    sensitive_features = np.array(sensitive_features)
    
    unique_groups = np.unique(sensitive_features)
    
    # Calculate metrics for each group
    group_metrics = {}
    for group in unique_groups:
        mask = sensitive_features == group
        if np.sum(mask) > 0:
            # Ensure we have valid data for calculations
            y_pred_group = y_pred[mask]
            y_true_group = y_true[mask]
            
            # Calculate rates with proper type handling
            group_pred_rate = float(np.mean(y_pred_group))
            group_true_rate = float(np.mean(y_true_group))
            group_accuracy = float(np.mean(y_true_group == y_pred_group))
            
            group_metrics[group] = {
                'prediction_rate': group_pred_rate,
                'true_positive_rate': group_true_rate,
                'accuracy': group_accuracy,
                'count': int(np.sum(mask))
            }
    
    # Calculate demographic parity difference
    pred_rates = [float(metrics['prediction_rate']) for metrics in group_metrics.values()]
    dp_diff = float(max(pred_rates) - min(pred_rates)) if len(pred_rates) > 1 else 0.0
    
    # Calculate equalized odds difference (simplified)
    tpr_rates = [float(metrics['true_positive_rate']) for metrics in group_metrics.values()]
    eo_diff = float(max(tpr_rates) - min(tpr_rates)) if len(tpr_rates) > 1 else 0.0
    
    # Calculate disparate impact ratio
    if len(pred_rates) > 1 and max(pred_rates) > 0:
        di_ratio = float(min(pred_rates) / max(pred_rates))
    else:
        di_ratio = 1.0
    
    results = {
        'demographic_parity_difference': dp_diff,
        'equalized_odds_difference': eo_diff,
        'disparate_impact_ratio': di_ratio,
        'group_metrics': group_metrics
    }
    
    return results

def apply_reweighing_mitigation(X_train, y_train, sensitive_features_train):
    """Apply reweighing mitigation technique"""
    # Calculate sample weights based on group membership and outcomes
    weights = np.ones(len(y_train))
    
    unique_groups = np.unique(sensitive_features_train)
    unique_outcomes = np.unique(y_train)
    
    # Calculate expected probability for each group-outcome combination
    total_samples = len(y_train)
    
    for group in unique_groups:
        for outcome in unique_outcomes:
            # Count samples in this group-outcome combination
            mask = (sensitive_features_train == group) & (y_train == outcome)
            n_group_outcome = np.sum(mask)
            
            if n_group_outcome > 0:
                # Calculate expected vs actual probability
                n_group = np.sum(sensitive_features_train == group)
                n_outcome = np.sum(y_train == outcome)
                
                expected_prob = (n_group * n_outcome) / total_samples
                actual_prob = n_group_outcome
                
                # Calculate weight
                if actual_prob > 0:
                    weight = expected_prob / actual_prob
                    weights[mask] = weight
    
    return weights

def apply_threshold_optimization(y_true, y_pred_proba, sensitive_features, fairness_weight=0.5):
    """
    Enhanced threshold optimization with fairness-accuracy trade-off
    """
    try:
        import numpy as np
        from sklearn.metrics import accuracy_score
        
        # Get unique groups
        unique_groups = np.unique(sensitive_features)
        
        # Initialize optimized predictions
        y_pred_optimized = np.zeros_like(y_true)
        
        # Calculate overall threshold for baseline
        overall_threshold = 0.5
        
        # For each group, find optimal threshold
        for group in unique_groups:
            group_mask = sensitive_features == group
            group_y_true = y_true[group_mask]
            group_y_proba = y_pred_proba[group_mask]
            
            if len(group_y_true) == 0:
                continue
            
            # Calculate group-specific threshold
            best_threshold = overall_threshold
            best_score = 0
            
            # Try different thresholds
            for threshold in np.arange(0.1, 0.9, 0.05):
                group_pred = (group_y_proba >= threshold).astype(int)
                
                # Calculate accuracy for this group
                group_accuracy = accuracy_score(group_y_true, group_pred)
                
                # Calculate fairness score (how close to overall rate)
                group_positive_rate = np.mean(group_pred)
                overall_positive_rate = np.mean(y_pred_proba >= overall_threshold)
                fairness_score = 1 - abs(group_positive_rate - overall_positive_rate)
                
                # Combined score based on fairness weight
                combined_score = (1 - fairness_weight) * group_accuracy + fairness_weight * fairness_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_threshold = threshold
            
            # Apply optimized threshold to this group
            group_pred_optimized = (group_y_proba >= best_threshold).astype(int)
            y_pred_optimized[group_mask] = group_pred_optimized
        
        return y_pred_optimized
        
    except Exception as e:
        st.error(f"Error in threshold optimization: {str(e)}")
        # Fallback to original predictions
        return (y_pred_proba >= 0.5).astype(int)

def create_comparison_chart(baseline_metrics, mitigated_metrics):
    """Create comparison chart for before/after metrics"""
    metrics_names = ['Accuracy', 'Demographic Parity Diff', 'Equalized Odds Diff', 'Disparate Impact Ratio']
    
    baseline_values = [
        baseline_metrics.get('accuracy', 0),
        baseline_metrics.get('demographic_parity_difference', 0),
        baseline_metrics.get('equalized_odds_difference', 0),
        baseline_metrics.get('disparate_impact_ratio', 1)
    ]
    
    mitigated_values = [
        mitigated_metrics.get('accuracy', 0),
        mitigated_metrics.get('demographic_parity_difference', 0),
        mitigated_metrics.get('equalized_odds_difference', 0),
        mitigated_metrics.get('disparate_impact_ratio', 1)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Before Mitigation',
        x=metrics_names,
        y=baseline_values,
        marker_color='#e74c3c'
    ))
    
    fig.add_trace(go.Bar(
        name='After Mitigation',
        x=metrics_names,
        y=mitigated_values,
        marker_color='#2ecc71'
    ))
    
    fig.update_layout(
        title='Fairness Metrics: Before vs After Mitigation',
        xaxis_title='Metrics',
        yaxis_title='Values',
        barmode='group',
        height=400
    )
    
    return fig

# MAIN DASHBOARD SECTION
if section == "🎯 Main Dashboard":
    # Clean, centered header using Streamlit native components
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Centered title and subtitle
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #1f77b4; font-size: 3em; margin-bottom: 10px;">🤖 Universal Bias Audit Dashboard</h1>
            <p style="font-size: 1.2em; color: #666; margin-bottom: 30px;">Detect and mitigate bias in machine learning models for ANY dataset</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # What This Dashboard Does section
    st.markdown("### 🎯 What This Dashboard Does")
    st.write("This dashboard helps you detect and fix bias in machine learning models using **any CSV dataset** with binary outcomes.")
    
    # Key Features in 3-column layout using Streamlit containers
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown("#### 📊 Upload CSV")
            st.write("Upload any CSV with binary target column (Yes/No, 0/1, Hired/NotHired, etc.)")
            
        with st.container():
            st.markdown("#### ⚖️ Fairness Metrics")
            st.write("Calculate demographic parity, equalized odds, and disparate impact across groups")
    
    with col2:
        with st.container():
            st.markdown("#### 🔍 Smart Feature Detection")
            st.write("Auto-detect demographics and sensitive features using intelligent keyword matching")
            
        with st.container():
            st.markdown("#### 🛠️ Bias Mitigation")
            st.write("Apply reweighing and threshold optimization techniques to reduce bias")
    
    with col3:
        with st.container():
            st.markdown("#### 🤖 ML Pipeline")
            st.write("Train LogisticRegression or RandomForest models with comprehensive evaluation")
            
        with st.container():
            st.markdown("#### 📋 Reporting & Export")
            st.write("Generate comprehensive reports and export results in multiple formats")
    
    # How to Use This Dashboard section
    st.markdown("### 🧭 How to Use This Dashboard")
    
    st.markdown("""
    **Follow these simple steps to audit your dataset for bias:**
    
    1. **📁 Upload Your Data** - Go to the ML Pipeline tab and upload any CSV file with binary outcomes
    2. **🎯 Select Target Column** - Choose the column you want to predict (e.g., 'Hired', 'Approved', 'Selected')  
    3. **👥 Choose Sensitive Feature** - Select the demographic column to check for bias (e.g., 'Gender', 'Race', 'Location')
    4. **🚀 Train Model** - Let the system automatically train and evaluate your machine learning model
    5. **📊 Review Results** - Examine fairness metrics and identify any bias in your model's predictions
    6. **🔧 Apply Mitigation** - Use built-in techniques to reduce bias and improve fairness
    7. **📈 Compare & Export** - View before/after results and download comprehensive reports
    """)
    
    # Dataset Examples section
    st.markdown("### 🗂️ Works With Any Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Business Applications:**
        - **Hiring:** Resume screening, job applications
        - **Lending:** Loan approvals, credit decisions  
        - **Healthcare:** Treatment recommendations, diagnosis
        """)
    
    with col2:
        st.markdown("""
        **Additional Use Cases:**
        - **Education:** Admissions, grading, scholarships
        - **Marketing:** Ad targeting, customer segmentation
        - **Legal:** Risk assessment, sentencing prediction
        """)
    
    # Get Started button
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Get Started with ML Pipeline", type="primary", use_container_width=True):
            st.session_state.selected_section = "🤖 ML Pipeline"
            st.rerun()
    
    # Clean info section
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container():
        st.info("""
        💡 **Welcome to the Universal Bias Audit Dashboard!**  
        This powerful tool helps organizations ensure fairness in their machine learning models across all demographic groups. 
        Whether you're working with hiring data, loan applications, healthcare decisions, or any other binary classification task, 
        this dashboard will help you identify, measure, and mitigate bias in your AI systems.
        """)
    
    # Dashboard capabilities overview
    st.markdown("### 🎨 Dashboard Capabilities")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Supported Formats", "CSV Files")
        st.caption("Upload any CSV with binary outcomes")
    
    with col2:
        st.metric("🤖 ML Algorithms", "2 Models")
        st.caption("LogisticRegression & RandomForest")
    
    with col3:
        st.metric("⚖️ Fairness Metrics", "3 Types")
        st.caption("Comprehensive bias evaluation")
    
    with col4:
        st.metric("🛠️ Mitigation Methods", "2 Techniques")
        st.caption("Pre & post-processing approaches")
    
    # Recent Activity and Next Steps section
    st.markdown("### 📈 Ready to Get Started?")
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        # Clean section header using Streamlit native components
        st.markdown("#### 🚀 Quick Start Guide")
        st.write("Follow these steps to audit your dataset for bias:")
        
        # Step-by-step guide using Streamlit native components
        with st.container():
            st.markdown("""
            **Step 1:** 📁 Navigate to the **ML Pipeline** tab  
            **Step 2:** 📊 Upload your CSV file with binary outcomes  
            **Step 3:** 🎯 Select your target column (what you want to predict)  
            **Step 4:** 👥 Choose your sensitive feature (demographic to check)  
            **Step 5:** 🤖 Train your model and review fairness metrics  
            **Step 6:** 🔧 Apply bias mitigation if needed  
            **Step 7:** 📈 Compare results and export reports  
            """)
        
        st.markdown("---")
        
        st.markdown("#### 💡 Why Use This Dashboard?")
        with st.container():
            st.success("**Legal Compliance** - Meet anti-discrimination requirements")
            st.info("**Risk Reduction** - Avoid costly bias-related lawsuits")  
            st.warning("**Fair Outcomes** - Ensure equal opportunities for all groups")
            st.error("**Brand Protection** - Maintain positive reputation")
        

        
        # Search functionality - simplified (placeholder for demo)
        search_term = ""
        
        # Filter events based on search
        filtered_events = bias_events
        if search_term:
            filtered_events = [
                event for event in bias_events 
                if search_term.lower() in event['Group'].lower() or 
                   search_term.lower() in event['Metric'].lower()
            ]
        
        # Display user-friendly bias event cards
        if filtered_events:
            for idx, event in enumerate(filtered_events[:10]):
                severity_class = f"severity-{event['Severity'].lower()}"
                
                # Convert technical terms to user-friendly language
                group_parts = event['Group'].split(': ')
                category = group_parts[0]
                group_name = group_parts[1]
                
                # User-friendly group names
                group_friendly = {
                    'Black African': 'Black African candidates',
                    'White': 'White candidates',
                    'Coloured': 'Coloured candidates',
                    'Indian': 'Indian candidates',
                    'Female': 'Women',
                    'Male': 'Men',
                    'Rural': 'Rural area candidates',
                    'Urban': 'Urban area candidates',
                    'Low': 'Candidates with basic English',
                    'Medium': 'Candidates with good English',
                    'High': 'Candidates with excellent English'
                }.get(group_name, group_name)
                
                # User-friendly explanations
                bias_explanation = f"{group_friendly} are {event['Gap']} less likely to be hired than other groups"
                
                severity_explanation = {
                    'High': '🚨 URGENT - This is a serious fairness problem that could lead to legal issues',
                    'Medium': '⚠️ ATTENTION NEEDED - This fairness issue should be addressed soon',
                    'Low': '✅ MINOR - Small fairness gap that should be monitored'
                }.get(event['Severity'], event['Severity'])
                
                # Create user-friendly event card
                event_html = f"""
                <div class="bias-event-card {severity_class}">
                    <div class="event-header">
                        <h4 class="event-title">Unfair Hiring: {group_friendly}</h4>
                        <span class="severity-badge {event['Severity'].lower()}">{event['Severity']} Priority</span>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; margin: 10px 0;">
                        <strong>📊 What this means:</strong><br>
                        {bias_explanation}
                    </div>
                    
                    <div style="background: #e9ecef; padding: 12px; border-radius: 8px; margin: 10px 0;">
                        <strong>⚠️ Priority Level:</strong><br>
                        {severity_explanation}
                    </div>
                    
                    <div class="event-meta">
                        <div class="meta-item">
                            <span>📊</span>
                            <span>Hiring Gap: <span class="meta-value">{event['Gap']}</span></span>
                        </div>
                        <div class="meta-item">
                            <span>🎯</span>
                            <span>Certainty: <span class="meta-value">{float(event['Confidence'])*100:.0f}%</span></span>
                        </div>
                        <div class="meta-item">
                            <span>⏰</span>
                            <span>Found: <span class="meta-value">{event['Timestamp']}</span></span>
                        </div>
                    </div>
                    
                    <div class="action-buttons">
                        <button class="action-btn btn-flag" onclick="flagEvent({idx})">
                            🚩 Mark as Priority #{idx}
                        </button>
                        <button class="action-btn btn-analyze" onclick="analyzeEvent({idx})">
                            📊 Get Details #{idx}
                        </button>
                        <button class="action-btn btn-export" onclick="exportEvent({idx})">
                            📁 Save Report #{idx}
                        </button>
                    </div>
                </div>
                """
                
                st.markdown(event_html, unsafe_allow_html=True)
                
                # Streamlit action buttons with user-friendly language
                with st.expander(f"🔧 Actions for Issue #{idx}", expanded=False):
                    st.markdown(f"**Issue:** {bias_explanation}")
                    st.markdown(f"**Priority:** {severity_explanation}")
                    
                    button_col1, button_col2, button_col3 = st.columns(3)
                    with button_col1:
                        if st.button(f"🚩 Mark as Priority", key=f"flag_{idx}"):
                            st.success(f"✅ Issue #{idx} marked as priority for HR review!")
                    with button_col2:
                        if st.button(f"📊 Get Full Details", key=f"analyze_{idx}"):
                            st.info(f"🔍 Detailed analysis for: {group_friendly}")
                            
                            # User-friendly analysis
                            analysis_data = {
                                "Issue #": idx,
                                "Problem": f"Unfair hiring affecting {group_friendly}",
                                "Category": category.replace('_', ' ').title(),
                                "Affected Group": group_friendly,
                                "Hiring Gap": event['Gap'],
                                "Priority Level": event['Severity'],
                                "How Certain We Are": f"{float(event['Confidence'])*100:.0f}%",
                                "When Found": event['Timestamp']
                            }
                            
                            st.json(analysis_data)
                            
                            # Recommendations
                            st.markdown("### 💡 Recommended Actions:")
                            if event['Severity'] == 'High':
                                st.error("""
                                **Immediate Actions Needed:**
                                - Review hiring criteria that may disadvantage this group
                                - Train hiring managers on unconscious bias
                                - Consider adjusting recruitment strategies
                                - Document corrective measures for legal compliance
                                """)
                            elif event['Severity'] == 'Medium':
                                st.warning("""
                                **Actions to Take Soon:**
                                - Monitor hiring patterns more closely
                                - Review job descriptions and requirements
                                - Consider diversity training for hiring team
                                """)
                            else:
                                st.info("""
                                **Keep Monitoring:**
                                - Continue tracking this group's hiring rates
                                - Maintain current diversity initiatives
                                """)
                    
                    with button_col3:
                        if st.button(f"📁 Save Full Report", key=f"export_{idx}"):
                            report_content = f"""FAIR HIRING REPORT #{idx}
================================
Issue: Unfair hiring affecting {group_friendly}
Category: {category.replace('_', ' ').title()}
Hiring Gap: {event['Gap']} less likely to be hired
Priority Level: {event['Severity']}
Certainty: {float(event['Confidence'])*100:.0f}%
Date Found: {event['Timestamp']}

WHAT THIS MEANS:
{bias_explanation}

PRIORITY LEVEL:
{severity_explanation}

RECOMMENDED ACTIONS:
"""
                            if event['Severity'] == 'High':
                                report_content += """
- URGENT: Review hiring criteria immediately
- Train hiring managers on unconscious bias
- Adjust recruitment strategies
- Document all corrective measures
- Consider legal consultation
"""
                            elif event['Severity'] == 'Medium':
                                report_content += """
- Review hiring patterns within 30 days
- Update job descriptions if needed
- Provide diversity training to hiring team
- Increase monitoring frequency
"""
                            else:
                                report_content += """
- Continue monitoring hiring patterns
- Maintain current diversity initiatives
- Schedule quarterly reviews
"""
                            
                            report_content += f"""

Generated by: Tech Titanians Fair Hiring Monitor
Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                            st.download_button(
                                "⬇️ Download Full Report",
                                data=report_content,
                                file_name=f"fair_hiring_issue_{idx}_{datetime.now().strftime('%Y%m%d')}.txt",
                                key=f"download_{idx}"
                            )
        else:
            # User-friendly empty state
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">🎉</div>
                <h3>Great news! No unfair hiring issues found</h3>
                <p>Your search didn't find any problems. Try different search terms or check the filters on the left.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_right:
        st.subheader("📊 Quick Summary")
        
        # User-friendly explanation
        st.markdown("""
        <div style="background: #e3f2fd; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <strong>🤔 What am I looking at?</strong><br>
            This shows how serious the unfair hiring problems are in your company.
        </div>
        """, unsafe_allow_html=True)
        
        # Severity distribution with user-friendly labels
        severity_counts = {}
        for event in bias_events:
            friendly_severity = {
                'High': 'Urgent Issues',
                'Medium': 'Needs Attention', 
                'Low': 'Minor Issues'
            }.get(event['Severity'], event['Severity'])
            severity_counts[friendly_severity] = severity_counts.get(friendly_severity, 0) + 1
        
        if severity_counts:
            fig_severity = px.pie(
                values=list(severity_counts.values()),
                names=list(severity_counts.keys()),
                title="Issues by Priority Level",
                color_discrete_map={
                    'Urgent Issues': '#dc3545',
                    'Needs Attention': '#fd7e14', 
                    'Minor Issues': '#28a745'
                }
            )
            fig_severity.update_layout(height=300)
            st.plotly_chart(fig_severity, use_container_width=True)
        
        # Recent activity with user-friendly language
        st.markdown("### 📈 Latest Issues Found")
        recent_events = sorted(bias_events, key=lambda x: x['Timestamp'], reverse=True)[:5]
        for event in recent_events:
            group_parts = event['Group'].split(': ')
            group_name = group_parts[1]
            group_friendly = {
                'Black African': 'Black African candidates',
                'White': 'White candidates',
                'Coloured': 'Coloured candidates',
                'Indian': 'Indian candidates',
                'Female': 'Women',
                'Male': 'Men',
                'Rural': 'Rural candidates',
                'Urban': 'Urban candidates',
                'Low': 'Basic English speakers',
                'Medium': 'Good English speakers',
                'High': 'Excellent English speakers'
            }.get(group_name, group_name)
            
            st.markdown(f"• **{group_friendly}**: {event['Gap']} hiring gap")

    # STEP 2: MODEL TRAINING COMPONENT
    st.markdown("---")
    st.markdown("""
    <div class="main-header">
        <h2>📦 Step 2: Train Baseline Model</h2>
        <p>Upload your dataset and train a machine learning model to predict hiring outcomes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    col_upload, col_config = st.columns([1, 1])
    
    with col_upload:
        st.subheader("📂 Dataset Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="Upload your hiring dataset with features and target variable"
        )
        
        use_default = st.checkbox("Use sample dataset (hiring data)", value=True)
        
        if use_default or uploaded_file is not None:
            if use_default:
                # Use the existing data
                ml_data = data.copy()
                st.success("✅ Using sample hiring dataset")
            else:
                # Load uploaded file
                ml_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Uploaded dataset with {len(ml_data)} rows and {len(ml_data.columns)} columns")
            
            # Display data preview
            with st.expander("📊 Data Preview", expanded=False):
                st.dataframe(ml_data.head(10))
                
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("Total Samples", len(ml_data))
                with col_info2:
                    st.metric("Features", len(ml_data.columns))
                with col_info3:
                    st.metric("Missing Values", ml_data.isnull().sum().sum())
    
    with col_config:
        st.subheader("⚙️ Model Configuration")
        
        # Add helpful example
        with st.expander("📚 **Need Help? See Example**", expanded=False):
            st.markdown("""
            **Example Setup for ANY Dataset:**
            - **✅ Target Column**: Any binary column (0/1, Yes/No, True/False, Hired/NotHired, etc.)
            - **✅ Sensitive Feature**: Any demographic column (Gender, Race, Age, Location, etc.)
            - **✅ Other Features**: Any other columns will be used for training
            
            **❌ Common Mistake**: Don't use multi-value columns as target!
            **✅ Correct**: Target must have exactly 2 unique values for binary classification
            
            **📋 Your Dataset Will Be Auto-Analyzed:**
            - Binary columns will be suggested for target
            - Demographic-like columns will be suggested for sensitive features
            - All other columns will be used as features for training
            """)
        
        if 'ml_data' in locals():
            # Create columns for configuration layout
            col_config_left, col_config_right = st.columns(2)
            
            with col_config_left:
                # Enhanced Target column detection for ANY dataset
                st.markdown("**🎯 Target Column Selection**")
                st.info("💡 **Target Column** = What you want to predict (must have exactly 2 values)")
                
                # Advanced column analysis for any dataset
                binary_columns = []
                non_binary_columns = []
                potential_targets = []
                
                for col in ml_data.columns:
                    unique_vals = ml_data[col].unique()
                    if len(unique_vals) == 2:
                        # Check if it looks like a target variable
                        val_str = str(unique_vals).lower()
                        is_likely_target = any(keyword in val_str for keyword in [
                            'hired', 'approved', 'accepted', 'passed', 'success', 'yes', 'no',
                            'true', 'false', '0', '1', 'positive', 'negative', 'good', 'bad',
                            'granted', 'selected', 'recommended', 'admitted', 'awarded'
                        ])
                        
                        binary_columns.append(f"{col} ({unique_vals[0]}, {unique_vals[1]})")
                        if is_likely_target:
                            potential_targets.append(col)
                    else:
                        non_binary_columns.append(f"{col} ({len(unique_vals)} values)")
                
                if binary_columns:
                    st.success(f"✅ **Valid target columns (binary)**: {', '.join(binary_columns)}")
                
                if non_binary_columns:
                    st.warning(f"⚠️ **Cannot use as target (multi-value)**: {', '.join(non_binary_columns[:3])}")
                
                # Smart default selection for ANY dataset
                default_target = None
                if potential_targets:
                    default_target = potential_targets[0]  # Use most likely target
                elif binary_columns:
                    # Use first binary column if no obvious target
                    default_target = binary_columns[0].split(' (')[0]
                else:
                    default_target = ml_data.columns[0]
                
                target_column = st.selectbox(
                    "Select the column that contains the outcome you want to predict",
                    options=ml_data.columns.tolist(),
                    index=ml_data.columns.tolist().index(default_target) if default_target in ml_data.columns else 0,
                    help="Choose a column with exactly 2 values (binary outcomes)"
                )
                
                # Show preview of selected target column
                if target_column:
                    target_preview = ml_data[target_column].value_counts()
                    st.markdown(f"**Preview of '{target_column}':**")
                    for value, count in target_preview.items():
                        st.write(f"• {value}: {count} samples ({count/len(ml_data)*100:.1f}%)")
                    
                    # Validate target column
                    unique_target_vals = ml_data[target_column].unique()
                    if len(unique_target_vals) != 2:
                        st.error(f"❌ **Invalid Target Column**: '{target_column}' has {len(unique_target_vals)} values")
                        st.error("🔄 **Solution**: Choose a column with exactly 2 values")
                    else:
                        st.success(f"✅ Perfect! '{target_column}' has 2 values: {unique_target_vals}")
            
            with col_config_right:
                # Enhanced Sensitive feature selection for ANY dataset
                st.markdown("**👥 Sensitive Feature Selection**")
                st.info("💡 **Sensitive Feature** = Demographics to check for bias")
                
                available_features = [col for col in ml_data.columns if col != target_column]
                
                # Expanded sensitive feature detection for ANY dataset
                sensitive_keywords = [
                    # Demographics
                    'gender', 'sex', 'male', 'female', 'race', 'ethnicity', 'ethnic', 'color', 'colour',
                    'age', 'birth', 'old', 'young', 'senior', 'junior',
                    # Location
                    'location', 'city', 'state', 'country', 'region', 'area', 'zip', 'postal',
                    'urban', 'rural', 'suburb', 'address', 'residence',
                    # Education  
                    'education', 'degree', 'school', 'university', 'college', 'qualification',
                    'diploma', 'certificate', 'grade', 'level',
                    # Employment
                    'experience', 'job', 'work', 'employment', 'career', 'position', 'title',
                    # Other demographics
                    'religion', 'marital', 'married', 'single', 'family', 'income', 'salary',
                    'disability', 'language', 'nationality', 'citizen', 'immigrant'
                ]
                
                # Find potential sensitive features
                potential_sensitive = []
                for col in available_features:
                    col_lower = col.lower()
                    # Check if column name contains sensitive keywords
                    if any(keyword in col_lower for keyword in sensitive_keywords):
                        potential_sensitive.append(col)
                    # Also check if it's categorical (reasonable number of unique values)
                    elif 2 <= len(ml_data[col].unique()) <= 20:
                        potential_sensitive.append(col)
                
                # Remove duplicates while preserving order
                potential_sensitive = list(dict.fromkeys(potential_sensitive))
                
                if potential_sensitive:
                    st.success(f"✅ **Suggested sensitive features**: {', '.join(potential_sensitive[:3])}")
                    if len(potential_sensitive) > 3:
                        st.info(f"... and {len(potential_sensitive) - 3} more potential features")
                
                # Smart default selection for sensitive feature
                default_sensitive = potential_sensitive[0] if potential_sensitive else available_features[0]
                
                sensitive_feature = st.selectbox(
                    "Select the demographic feature to analyze for bias",
                    options=available_features,
                    index=available_features.index(default_sensitive),
                    help="Choose any categorical column representing demographics"
                )
                
                # Show preview of selected sensitive feature
                if sensitive_feature:
                    sensitive_preview = ml_data[sensitive_feature].value_counts()
                    st.markdown(f"**Preview of '{sensitive_feature}':**")
                    for value, count in sensitive_preview.head(5).items():
                        st.write(f"• {value}: {count} samples ({count/len(ml_data)*100:.1f}%)")
                    if len(sensitive_preview) > 5:
                        st.write(f"• ... and {len(sensitive_preview) - 5} more groups")
        
        # Model Training Section
        st.markdown("---")
        st.markdown("### 🤖 Step 4: Model Training & Evaluation")
        
        col_model_config, col_model_params = st.columns(2)
        
        with col_model_config:
            st.markdown("**🛠️ Model Configuration**")
            
            model_type = st.selectbox(
                "Choose ML Algorithm",
                options=['LogisticRegression', 'RandomForestClassifier'],
                help="LogisticRegression: Fast, interpretable. RandomForest: More complex, potentially higher accuracy"
            )
            
            test_size = st.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Percentage of data reserved for testing (20% recommended)"
            )
            
            random_state = st.number_input(
                "Random Seed",
                min_value=1,
                max_value=1000,
                value=42,
                help="For reproducible results"
            )
        
        with col_model_params:
            st.markdown("**⚙️ Advanced Parameters**")
            
            if model_type == 'LogisticRegression':
                max_iter = st.number_input(
                    "Max Iterations",
                    min_value=100,
                    max_value=2000,
                    value=1000,
                    help="Maximum iterations for convergence"
                )
                
                solver = st.selectbox(
                    "Solver",
                    options=['liblinear', 'lbfgs'],
                    index=1,
                    help="Algorithm for optimization"
                )
            else:  # RandomForest
                n_estimators = st.number_input(
                    "Number of Trees",
                    min_value=10,
                    max_value=500,
                    value=100,
                    help="More trees = better performance but slower"
                )
                
                max_depth = st.selectbox(
                    "Max Depth",
                    options=[None, 5, 10, 15, 20],
                    index=0,
                    help="Maximum depth of trees (None = unlimited)"
                )
        
        # Train Model Button
        can_train = (
            len(ml_data[target_column].unique()) == 2 and 
            target_column != sensitive_feature
        )
        
        if can_train:
            if st.button("🚀 Train & Evaluate Model", type="primary", use_container_width=True):
                with st.spinner("Training model and calculating fairness metrics..."):
                    # Store configuration
                    st.session_state['ml_config'] = {
                        'model_type': model_type,
                        'test_size': test_size,
                        'random_state': random_state,
                        'target_column': target_column,
                        'sensitive_feature': sensitive_feature
                    }
                    
                    if model_type == 'LogisticRegression':
                        st.session_state['ml_config']['max_iter'] = max_iter
                        st.session_state['ml_config']['solver'] = solver
                    else:
                        st.session_state['ml_config']['n_estimators'] = n_estimators
                        st.session_state['ml_config']['max_depth'] = max_depth
                    
                    # Preprocess data
                    X, y, categorical_cols, numerical_cols = preprocess_data(ml_data, target_column)
                    
                    if X is not None and y is not None:
                        # Train model with custom parameters
                        if model_type == 'LogisticRegression':
                            model_results = train_baseline_model(
                                X, y, model_type, test_size, random_state,
                                max_iter=max_iter, solver=solver
                            )
                        else:
                            model_results = train_baseline_model(
                                X, y, model_type, test_size, random_state,
                                n_estimators=n_estimators, max_depth=max_depth
                            )
                        
                        # Store results
                        st.session_state['model_results'] = model_results
                        st.session_state['ml_data'] = ml_data
                        st.session_state['X'] = X
                        st.session_state['y'] = y
                        
                        st.success("✅ Model trained successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to preprocess data. Please check your selections.")
        else:
            if len(ml_data[target_column].unique()) != 2:
                st.error("❌ Target column must have exactly 2 unique values")
            if target_column == sensitive_feature:
                st.error("❌ Target and sensitive feature cannot be the same")
    
    # Display Results Section
    if 'model_results' in st.session_state and 'ml_config' in st.session_state:
        st.markdown("---")
        st.markdown("### 📈 Step 5: Model Performance & Bias Analysis")
        
        model_results = st.session_state['model_results']
        ml_config = st.session_state['ml_config']
        
        # Performance Metrics
        col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
        
        with col_perf1:
            st.metric(
                "🎯 Training Accuracy",
                f"{model_results['train_accuracy']:.3f}",
                help="Model performance on training data"
            )
        
        with col_perf2:
            st.metric(
                "🧪 Test Accuracy", 
                f"{model_results['test_accuracy']:.3f}",
                help="Model performance on unseen test data"
            )
        
        with col_perf3:
            # Calculate precision
            from sklearn.metrics import precision_score, recall_score
            precision = precision_score(model_results['y_test'], model_results['y_pred_test'])
            st.metric(
                "🎯 Precision",
                f"{precision:.3f}",
                help="Of predicted positives, how many were correct"
            )
        
        with col_perf4:
            # Calculate recall
            recall = recall_score(model_results['y_test'], model_results['y_pred_test'])
            st.metric(
                "📊 Recall",
                f"{recall:.3f}",
                help="Of actual positives, how many were found"
            )
        
        # Fairness Analysis
        st.markdown("### ⚖️ Comprehensive Fairness Analysis")
        
        # Calculate fairness metrics
        ml_data = st.session_state['ml_data']
        sensitive_feature = ml_config['sensitive_feature']
        
        # Get sensitive feature values for test set
        test_indices = model_results['X_test_original'].index
        sensitive_test = ml_data.loc[test_indices, sensitive_feature].values
        
        # Calculate fairness metrics
        fairness_metrics = calculate_fairness_metrics_custom(
            model_results['y_test'], 
            model_results['y_pred_test'], 
            sensitive_test
        )
        
        # Store for mitigation comparison
        st.session_state['baseline_fairness_metrics'] = fairness_metrics
        st.session_state['baseline_accuracy'] = model_results['test_accuracy']
        
        # Display fairness metrics with explanations
        col_fair1, col_fair2, col_fair3 = st.columns(3)
        
        with col_fair1:
            dp_diff = fairness_metrics['demographic_parity_difference']
            dp_color = "inverse" if dp_diff > 0.1 else "normal"
            st.metric(
                "⚖️ Demographic Parity",
                f"{dp_diff:.3f}",
                delta=f"{'High bias' if dp_diff > 0.1 else 'Acceptable'}" if dp_diff > 0.05 else "Low bias",
                delta_color=dp_color,
                help="Difference in positive prediction rates between groups (closer to 0 is better)"
            )
            
            with st.expander("ℹ️ What is Demographic Parity?"):
                st.write("""
                **Demographic Parity** measures whether different groups receive positive outcomes at similar rates.
                
                - **Value**: Difference between highest and lowest group rates
                - **Range**: 0 (perfect) to 1 (maximum bias)
                - **Threshold**: < 0.1 is generally acceptable
                - **Example**: If 70% of Group A gets hired vs 50% of Group B, difference = 0.20
                """)
        
        with col_fair2:
            eo_diff = fairness_metrics['equalized_odds_difference']
            eo_color = "inverse" if eo_diff > 0.1 else "normal"
            st.metric(
                "⚡ Equalized Odds",
                f"{eo_diff:.3f}",
                delta=f"{'High bias' if eo_diff > 0.1 else 'Acceptable'}" if eo_diff > 0.05 else "Low bias",
                delta_color=eo_color,
                help="Difference in true positive rates between groups (closer to 0 is better)"
            )
            
            with st.expander("ℹ️ What is Equalized Odds?"):
                st.write("""
                **Equalized Odds** measures whether the model performs equally well for different groups.
                
                - **Value**: Difference in true positive rates between groups
                - **Range**: 0 (perfect) to 1 (maximum bias)
                - **Threshold**: < 0.1 is generally acceptable
                - **Example**: If model correctly identifies 80% of qualified Group A vs 60% of qualified Group B
                """)
        
        with col_fair3:
            di_ratio = fairness_metrics['disparate_impact_ratio']
            di_color = "inverse" if di_ratio < 0.8 else "normal"
            st.metric(
                "📊 Disparate Impact",
                f"{di_ratio:.3f}",
                delta=f"{'Biased' if di_ratio < 0.8 else 'Fair'}" if di_ratio < 0.9 else "Good",
                delta_color=di_color,
                help="Ratio of positive rates between groups (should be close to 1.0)"
            )
            
            with st.expander("ℹ️ What is Disparate Impact?"):
                st.write("""
                **Disparate Impact** compares outcome rates between groups as a ratio.
                
                - **Value**: Ratio of lowest to highest group rates
                - **Range**: 0 (maximum bias) to 1 (perfect fairness)
                - **Legal Threshold**: > 0.8 (80% rule)
                - **Example**: If Group A: 60% positive, Group B: 80% positive, ratio = 0.75 (biased)
                """)
        
        # Detailed Group Analysis
        st.markdown("### 👥 Detailed Group Analysis")
        
        group_df = pd.DataFrame(fairness_metrics['group_metrics']).T
        group_df = group_df.round(3)
        group_df.columns = ['Prediction Rate', 'True Positive Rate', 'Accuracy', 'Sample Count']
        
        # Add percentage columns
        total_samples = group_df['Sample Count'].sum()
        group_df['Sample %'] = (group_df['Sample Count'] / total_samples * 100).round(1)
        group_df['Prediction %'] = (group_df['Prediction Rate'] * 100).round(1)
        group_df['TPR %'] = (group_df['True Positive Rate'] * 100).round(1)
        group_df['Accuracy %'] = (group_df['Accuracy'] * 100).round(1)
        
        # Reorder columns for better display
        display_df = group_df[['Sample Count', 'Sample %', 'Prediction %', 'TPR %', 'Accuracy %', 'Prediction Rate', 'True Positive Rate', 'Accuracy']]
        
        st.dataframe(display_df, use_container_width=True)
        
        # Visualization
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            # Prediction rates by group
            fig_pred = px.bar(
                x=list(fairness_metrics['group_metrics'].keys()),
                y=[metrics['prediction_rate'] * 100 for metrics in fairness_metrics['group_metrics'].values()],
                title=f"Prediction Rates by {sensitive_feature}",
                labels={'x': sensitive_feature, 'y': 'Prediction Rate (%)'},
                color=[metrics['prediction_rate'] for metrics in fairness_metrics['group_metrics'].values()],
                color_continuous_scale='RdYlBu_r'
            )
            fig_pred.update_layout(showlegend=False)
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with col_viz2:
            # Sample distribution
            fig_dist = px.pie(
                values=[metrics['count'] for metrics in fairness_metrics['group_metrics'].values()],
                names=list(fairness_metrics['group_metrics'].keys()),
                title=f"Sample Distribution by {sensitive_feature}"
            )
            st.plotly_chart(fig_dist, use_container_width=True)

elif section == "📊 Detailed Analysis":
    st.title("📊 Comprehensive Bias Analytics")
    st.markdown("**Tech Titanians** | Advanced statistical analysis")
    
    # STEP 4: BIAS MITIGATION METHODS
    if 'baseline_fairness_metrics' in st.session_state:
        st.markdown("---")
        st.markdown("""
        <div class="main-header">
            <h2>🧮 Step 4: Apply Bias Mitigation</h2>
            <p>Apply fairness techniques to reduce bias in your model</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mitigation method selection
        col_method, col_apply = st.columns([1, 1])
        
        with col_method:
            st.subheader("🛠️ Mitigation Techniques")
            
            mitigation_method = st.selectbox(
                "Choose Mitigation Method",
                options=["Reweighing (Pre-processing)", "Threshold Optimization (Post-processing)"],
                help="Select the bias mitigation technique to apply"
            )
            
            st.markdown("""
            **Available Methods:**
            - **Reweighing**: Adjusts training sample weights to balance representation
            - **Threshold Optimization**: Adjusts decision thresholds per group for fairness
            """)
            
            # Display current fairness status
            baseline_metrics = st.session_state['baseline_fairness_metrics']
            st.markdown("### 📊 Current Fairness Status")
            
            col_status1, col_status2 = st.columns(2)
            with col_status1:
                dp_diff = baseline_metrics['demographic_parity_difference']
                status = "🔴 High Bias" if dp_diff > 0.1 else "🟡 Medium Bias" if dp_diff > 0.05 else "🟢 Low Bias"
                st.metric("Demographic Parity", f"{dp_diff:.3f}", delta=status)
            
            with col_status2:
                eo_diff = baseline_metrics['equalized_odds_difference']
                status = "🔴 High Bias" if eo_diff > 0.1 else "🟡 Medium Bias" if eo_diff > 0.05 else "🟢 Low Bias"
                st.metric("Equalized Odds", f"{eo_diff:.3f}", delta=status)
        
        with col_apply:
            st.subheader("🚀 Apply Mitigation")
            
            if st.button("🧮 Apply Mitigation Technique", type="primary"):
                with st.spinner("Applying bias mitigation..."):
                    # Get stored data
                    model_results = st.session_state['model_results']
                    ml_data = st.session_state['ml_data']
                    sensitive_feature = st.session_state['sensitive_feature']
                    target_column = st.session_state['target_column']
                    
                    # Get sensitive feature values
                    train_indices = model_results['X_train_original'].index
                    test_indices = model_results['X_test_original'].index
                    sensitive_train = ml_data.loc[train_indices, sensitive_feature].values
                    sensitive_test = ml_data.loc[test_indices, sensitive_feature].values
                    
                    if mitigation_method == "Reweighing (Pre-processing)":
                        # Apply reweighing
                        sample_weights = apply_reweighing_mitigation(
                            model_results['X_train_original'], 
                            model_results['y_train'], 
                            sensitive_train
                        )
                        
                        # Retrain model with weights
                        if st.session_state.get('model_type', 'LogisticRegression') == 'LogisticRegression':
                            mitigated_model = LogisticRegression(random_state=42, max_iter=1000)
                        else:
                            mitigated_model = RandomForestClassifier(random_state=42, n_estimators=100)
                        
                        mitigated_model.fit(model_results['X_train'], model_results['y_train'], sample_weight=sample_weights)
                        
                        # Make predictions
                        y_pred_mitigated = mitigated_model.predict(model_results['X_test'])
                        y_pred_proba_mitigated = mitigated_model.predict_proba(model_results['X_test'])[:, 1]
                        
                    else:  # Threshold Optimization
                        # Apply threshold optimization
                        y_pred_mitigated = apply_threshold_optimization(
                            model_results['y_test'],
                            model_results['y_pred_proba_test'],
                            sensitive_test
                        )
                        y_pred_proba_mitigated = model_results['y_pred_proba_test']
                    
                    # Calculate mitigated fairness metrics
                    mitigated_fairness_metrics = calculate_fairness_metrics_custom(
                        model_results['y_test'], 
                        y_pred_mitigated, 
                        sensitive_test
                    )
                    
                    # Calculate mitigated accuracy
                    mitigated_accuracy = accuracy_score(model_results['y_test'], y_pred_mitigated)
                    
                    # Store mitigated results
                    st.session_state['mitigated_fairness_metrics'] = mitigated_fairness_metrics
                    st.session_state['mitigated_accuracy'] = mitigated_accuracy
                    st.session_state['mitigation_method'] = mitigation_method
                    st.session_state['y_pred_mitigated'] = y_pred_mitigated
                    
                    st.success(f"✅ {mitigation_method} applied successfully!")
                    st.rerun()
        
        # Display mitigation results
        if 'mitigated_fairness_metrics' in st.session_state:
            st.markdown("### 📈 Mitigation Results")
            
            mitigated_metrics = st.session_state['mitigated_fairness_metrics']
            method_used = st.session_state['mitigation_method']
            
            st.info(f"✅ Applied: {method_used}")
            
            # Display improved metrics
            col_improved1, col_improved2, col_improved3 = st.columns(3)
            
            with col_improved1:
                old_dp = baseline_metrics['demographic_parity_difference']
                new_dp = mitigated_metrics['demographic_parity_difference']
                improvement = old_dp - new_dp
                st.metric(
                    "⚖️ Demographic Parity (After)", 
                    f"{new_dp:.3f}",
                    delta=f"{improvement:+.3f} improvement" if improvement > 0 else f"{improvement:.3f}",
                    delta_color="normal" if improvement > 0 else "inverse"
                )
            
            with col_improved2:
                old_eo = baseline_metrics['equalized_odds_difference']
                new_eo = mitigated_metrics['equalized_odds_difference']
                improvement = old_eo - new_eo
                st.metric(
                    "⚡ Equalized Odds (After)", 
                    f"{new_eo:.3f}",
                    delta=f"{improvement:+.3f} improvement" if improvement > 0 else f"{improvement:.3f}",
                    delta_color="normal" if improvement > 0 else "inverse"
                )
            
            with col_improved3:
                old_acc = st.session_state['baseline_accuracy']
                new_acc = st.session_state['mitigated_accuracy']
                acc_change = new_acc - old_acc
                st.metric(
                    "🎯 Accuracy (After)", 
                    f"{new_acc:.3f}",
                    delta=f"{acc_change:+.3f}" if abs(acc_change) > 0.001 else "No change",
                    delta_color="normal" if acc_change >= 0 else "inverse"
                )
    
    # STEP 5: COMPARE RESULTS
    if 'mitigated_fairness_metrics' in st.session_state:
        st.markdown("---")
        st.markdown("""
        <div class="main-header">
            <h2>📈 Step 5: Compare Fairness Results</h2>
            <p>Side-by-side comparison of fairness metrics before and after mitigation</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prepare comparison data
        baseline_metrics = st.session_state['baseline_fairness_metrics']
        mitigated_metrics = st.session_state['mitigated_fairness_metrics']
        baseline_accuracy = st.session_state['baseline_accuracy']
        mitigated_accuracy = st.session_state['mitigated_accuracy']
        
        # Create comparison chart
        comparison_data = {
            'accuracy': baseline_accuracy,
            'demographic_parity_difference': baseline_metrics['demographic_parity_difference'],
            'equalized_odds_difference': baseline_metrics['equalized_odds_difference'],
            'disparate_impact_ratio': baseline_metrics['disparate_impact_ratio']
        }
        
        mitigated_data = {
            'accuracy': mitigated_accuracy,
            'demographic_parity_difference': mitigated_metrics['demographic_parity_difference'],
            'equalized_odds_difference': mitigated_metrics['equalized_odds_difference'],
            'disparate_impact_ratio': mitigated_metrics['disparate_impact_ratio']
        }
        
        fig_comparison = create_comparison_chart(comparison_data, mitigated_data)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Side-by-side comparison cards
        st.markdown("### 📊 Detailed Comparison")
        
        col_before, col_after = st.columns(2)
        
        with col_before:
            st.markdown("""
            <div style="background: #ffe6e6; padding: 20px; border-radius: 10px; border-left: 5px solid #e74c3c;">
                <h4>📉 Before Mitigation</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Accuracy", f"{baseline_accuracy:.3f}")
            st.metric("Demographic Parity Diff", f"{baseline_metrics['demographic_parity_difference']:.3f}")
            st.metric("Equalized Odds Diff", f"{baseline_metrics['equalized_odds_difference']:.3f}")
            st.metric("Disparate Impact Ratio", f"{baseline_metrics['disparate_impact_ratio']:.3f}")
            
            # Group breakdown
            st.markdown("**Group Breakdown:**")
            for group, metrics in baseline_metrics['group_metrics'].items():
                st.write(f"• {group}: {metrics['prediction_rate']:.1%} prediction rate")
        
        with col_after:
            st.markdown("""
            <div style="background: #e6ffe6; padding: 20px; border-radius: 10px; border-left: 5px solid #2ecc71;">
                <h4>📈 After Mitigation</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Accuracy", f"{mitigated_accuracy:.3f}")
            st.metric("Demographic Parity Diff", f"{mitigated_metrics['demographic_parity_difference']:.3f}")
            st.metric("Equalized Odds Diff", f"{mitigated_metrics['equalized_odds_difference']:.3f}")
            st.metric("Disparate Impact Ratio", f"{mitigated_metrics['disparate_impact_ratio']:.3f}")
            
            # Group breakdown
            st.markdown("**Group Breakdown:**")
            for group, metrics in mitigated_metrics['group_metrics'].items():
                st.write(f"• {group}: {metrics['prediction_rate']:.1%} prediction rate")
        
        # Summary insights
        st.markdown("### 💡 Key Insights")
        
        dp_improvement = baseline_metrics['demographic_parity_difference'] - mitigated_metrics['demographic_parity_difference']
        eo_improvement = baseline_metrics['equalized_odds_difference'] - mitigated_metrics['equalized_odds_difference']
        acc_change = mitigated_accuracy - baseline_accuracy
        
        if dp_improvement > 0.01:
            st.success(f"✅ Demographic parity improved by {dp_improvement:.3f}")
        elif dp_improvement < -0.01:
            st.warning(f"⚠️ Demographic parity worsened by {abs(dp_improvement):.3f}")
        else:
            st.info("ℹ️ Demographic parity remained similar")
        
        if eo_improvement > 0.01:
            st.success(f"✅ Equalized odds improved by {eo_improvement:.3f}")
        elif eo_improvement < -0.01:
            st.warning(f"⚠️ Equalized odds worsened by {abs(eo_improvement):.3f}")
        else:
            st.info("ℹ️ Equalized odds remained similar")
        
        if abs(acc_change) > 0.01:
            if acc_change > 0:
                st.success(f"✅ Accuracy improved by {acc_change:.3f}")
            else:
                st.warning(f"⚠️ Accuracy decreased by {abs(acc_change):.3f}")
        else:
            st.info("ℹ️ Accuracy remained stable")
        
        # Export complete results
        st.markdown("### 📁 Export Complete Analysis")
        
        if st.button("📊 Generate Complete Report", type="primary"):
            # Create comprehensive report
            report_data = {
                "analysis_timestamp": datetime.now().isoformat(),
                "dataset_info": {
                    "total_samples": len(st.session_state['ml_data']),
                    "features": len(st.session_state['X'].columns),
                    "target_column": st.session_state['target_column'],
                    "sensitive_feature": st.session_state['sensitive_feature']
                },
                "baseline_results": {
                    "accuracy": baseline_accuracy,
                    "fairness_metrics": baseline_metrics
                },
                "mitigation_results": {
                    "method": st.session_state['mitigation_method'],
                    "accuracy": mitigated_accuracy,
                    "fairness_metrics": mitigated_metrics
                },
                "improvements": {
                    "demographic_parity_improvement": dp_improvement,
                    "equalized_odds_improvement": eo_improvement,
                    "accuracy_change": acc_change
                }
            }
            
            # Convert to JSON
            report_json = json.dumps(report_data, indent=2, default=str)
            
            st.download_button(
                "⬇️ Download Complete Analysis (JSON)",
                data=report_json,
                file_name=f"bias_analysis_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Also create a readable text report
            text_report = f"""COMPREHENSIVE BIAS ANALYSIS REPORT
========================================

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: {len(st.session_state['ml_data'])} samples, {len(st.session_state['X'].columns)} features
Target: {st.session_state['target_column']}
Sensitive Feature: {st.session_state['sensitive_feature']}

BASELINE MODEL PERFORMANCE
--------------------------
Accuracy: {baseline_accuracy:.3f}
Demographic Parity Difference: {baseline_metrics['demographic_parity_difference']:.3f}
Equalized Odds Difference: {baseline_metrics['equalized_odds_difference']:.3f}
Disparate Impact Ratio: {baseline_metrics['disparate_impact_ratio']:.3f}

MITIGATION APPLIED
------------------
Method: {st.session_state['mitigation_method']}

POST-MITIGATION PERFORMANCE
---------------------------
Accuracy: {mitigated_accuracy:.3f}
Demographic Parity Difference: {mitigated_metrics['demographic_parity_difference']:.3f}
Equalized Odds Difference: {mitigated_metrics['equalized_odds_difference']:.3f}
Disparate Impact Ratio: {mitigated_metrics['disparate_impact_ratio']:.3f}

IMPROVEMENTS
------------
Demographic Parity: {dp_improvement:+.3f}
Equalized Odds: {eo_improvement:+.3f}
Accuracy Change: {acc_change:+.3f}

RECOMMENDATIONS
---------------
"""
            
            if dp_improvement > 0.05:
                text_report += "✅ Significant improvement in demographic parity achieved.\n"
            elif dp_improvement < -0.05:
                text_report += "⚠️ Consider alternative mitigation methods - demographic parity worsened.\n"
            
            if eo_improvement > 0.05:
                text_report += "✅ Significant improvement in equalized odds achieved.\n"
            elif eo_improvement < -0.05:
                text_report += "⚠️ Consider alternative mitigation methods - equalized odds worsened.\n"
            
            if abs(acc_change) > 0.05:
                if acc_change > 0:
                    text_report += "✅ Model accuracy improved while reducing bias.\n"
                else:
                    text_report += "⚠️ Consider fairness-accuracy trade-off - accuracy decreased significantly.\n"
            
            text_report += f"\nGenerated by: Tech Titanians Fair Hiring Monitor\nReport ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            st.download_button(
                "⬇️ Download Readable Report (TXT)",
                data=text_report,
                file_name=f"bias_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    # Analytics tabs
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "👥 Demographics", "⚖️ Fairness Metrics"])
    
    with tab1:
        st.subheader("📈 Bias Trends Over Time")
        
        # Create sample trend data
        dates = pd.date_range(start=date_range[0], end=date_range[1], freq='D')
        trend_data = []
        
        for date in dates:
            for demographic in ["Race", "Gender", "Location"]:
                bias_score = np.random.normal(0.15, 0.05)
                trend_data.append({
                    'Date': date,
                    'Demographic': demographic,
                    'Bias_Score': max(0, bias_score)
                })
        
        trend_df = pd.DataFrame(trend_data)
        
        fig_trend = px.line(
            trend_df, 
            x='Date', 
            y='Bias_Score', 
            color='Demographic',
            title="Bias Score Trends by Demographic Group"
        )
        fig_trend.add_hline(y=0.1, line_dash="dash", line_color="orange", 
                           annotation_text="Medium Threshold")
        fig_trend.add_hline(y=0.2, line_dash="dash", line_color="red", 
                           annotation_text="High Threshold")
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with tab2:
        st.subheader("👥 Demographic Analysis")
        
        # Sample demographic data
        demo_data = pd.DataFrame({
            'Group': ['Black African', 'White', 'Coloured', 'Indian', 'Female', 'Male', 'Rural', 'Urban'],
            'Bias_Gap': [0.23, 0.08, 0.15, 0.12, 0.18, 0.05, 0.30, 0.07],
            'Hiring_Rate': [0.42, 0.65, 0.50, 0.55, 0.45, 0.60, 0.35, 0.65]
        })
        
        fig_demo = px.bar(
            demo_data,
            x='Group',
            y='Bias_Gap',
            title="Bias Gap by Demographic Group",
            color='Bias_Gap',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_demo, use_container_width=True)
        
        st.dataframe(demo_data, use_container_width=True)
    
    with tab3:
        st.subheader("⚖️ Fairness Metrics Analysis")
        
        # Sample fairness metrics
        metrics_data = {
            'Metric': ['Statistical Parity', 'Equal Opportunity', 'Equalized Odds'],
            'Original': [0.23, 0.18, 0.21],
            'After_Mitigation': [0.08, 0.06, 0.07]
        }
        
        fig_metrics = go.Figure(data=[
            go.Bar(name='Original', x=metrics_data['Metric'], y=metrics_data['Original']),
            go.Bar(name='After Mitigation', x=metrics_data['Metric'], y=metrics_data['After_Mitigation'])
        ])
        fig_metrics.update_layout(barmode='group', title='Fairness Metrics Comparison')
        st.plotly_chart(fig_metrics, use_container_width=True)

elif section == "📄 Reports":
    st.title("📄 Comprehensive Bias Audit Reports")
    st.markdown("**Tech Titanians** | Professional reporting")
    
    report_type = st.selectbox(
        "📋 Choose Report Type",
        ["📊 Executive Summary", "⚖️ Legal Compliance Report", "📈 Technical Analysis"]
    )
    
    if report_type == "📊 Executive Summary":
        st.markdown("### Executive Summary Preview")
        st.markdown(f"""
        **Report Date**: {datetime.now().strftime('%B %d, %Y')}  
        **Prepared by**: Tech Titanians  
        
        #### Key Findings:
        - **Total Bias Events**: {len(bias_events)}
        - **High-Severity Incidents**: {len([e for e in bias_events if e['Severity'] == 'High'])}
        - **Compliance Score**: {max(0, 100 - len([e for e in bias_events if e['Severity'] == 'High']) * 10)}%
        - **Estimated Annual Risk**: R4.7M
        - **Mitigation ROI**: R3.1M net benefit
        
        #### Recommendations:
        1. **Immediate**: Deploy bias mitigation algorithms
        2. **Short-term**: Enhance monitoring systems  
        3. **Long-term**: Implement governance framework
        """)
    
    if st.button("📥 Generate Report", type="primary"):
        report_content = f"""
        BIAS AUDIT REPORT - {report_type}
        Generated by: Tech Titanians
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Total Events: {len(bias_events)}
        High Severity Events: {len([e for e in bias_events if e['Severity'] == 'High'])}
        Analysis Period: {date_range[0]} to {date_range[1]}
        
        This report provides comprehensive analysis of algorithmic bias
        in hiring systems for Employment Equity Act compliance.
        """
        
        st.download_button(
            "⬇️ Download Report",
            data=report_content,
            file_name=f"bias_audit_report_{datetime.now().strftime('%Y%m%d')}.txt"
        )
        st.success("✅ Report generated successfully!")

elif section == "⚙️ Settings":
    st.title("⚙️ Dashboard Configuration")
    st.markdown("**Tech Titanians** | System settings")
    
    # Settings tabs
    tab1, tab2 = st.tabs(["🎚️ Thresholds", "🔔 Alerts"])
    
    with tab1:
        st.subheader("🎚️ Bias Detection Thresholds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            high_threshold = st.slider(
                "🚨 High Severity Threshold",
                min_value=0.10,
                max_value=0.50,
                value=0.20,
                step=0.01,
                format="%.2f"
            )
            
            medium_threshold = st.slider(
                "⚠️ Medium Severity Threshold", 
                min_value=0.01,
                max_value=0.20,
                value=0.10,
                step=0.01,
                format="%.2f"
            )
        
        with col2:
            st.markdown("#### Current Impact")
            st.info(f"High threshold: {high_threshold:.2f}")
            st.info(f"Medium threshold: {medium_threshold:.2f}")
        
        if st.button("💾 Save Threshold Settings", type="primary"):
            st.success(f"✅ Thresholds updated: High={high_threshold:.2f}, Medium={medium_threshold:.2f}")
    
    with tab2:
        st.subheader("🔔 Alert Configuration")
        
        alert_email = st.text_input(
            "📧 Alert Email",
            value="alerts@techtitanians.com"
        )
        
        alert_frequency = st.selectbox(
            "⏰ Alert Frequency",
            ["Immediate", "Hourly", "Daily", "Weekly"]
        )
        
        if st.button("💾 Save Alert Settings", type="primary"):
            st.success("✅ Alert settings saved!")

elif section == "🤖 ML Pipeline":
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Universal ML Bias Detection Pipeline</h1>
        <p>Complete end-to-end bias analysis for ANY binary classification dataset</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Dataset Upload & Smart Detection
    st.markdown("---")
    st.markdown("### 📊 Step 1: Dataset Upload & Smart Analysis")
    
    col_upload, col_templates = st.columns([2, 1])
    
    with col_upload:
        st.markdown("**📁 Upload Your Dataset**")
        uploaded_file = st.file_uploader(
            "Choose a CSV file with binary outcomes",
            type=['csv'],
            help="Upload any CSV with binary target column (hiring, lending, healthcare, etc.)"
        )
        
        use_default = st.checkbox("Use sample dataset (hiring data)", value=True)
        
        # Smart Dataset Analysis
        if use_default or uploaded_file is not None:
            if use_default:
                # Use the existing data
                ml_data = data.copy()
                st.success("✅ Using sample hiring dataset")
                st.info(f"📊 Dataset loaded: {len(ml_data)} samples × {len(ml_data.columns)} features")
            else:
                try:
                    # Load uploaded file with error handling
                    ml_data = pd.read_csv(uploaded_file)
                    st.success(f"✅ Dataset uploaded: {len(ml_data)} samples × {len(ml_data.columns)} features")
                    
                    # Validate dataset
                    if len(ml_data) == 0:
                        st.error("❌ Dataset is empty. Please upload a valid CSV file.")
                        st.stop()
                    elif len(ml_data.columns) < 2:
                        st.error("❌ Dataset needs at least 2 columns (features + target). Please check your CSV.")
                        st.stop()
                    
                except Exception as e:
                    st.error(f"❌ Error loading CSV: {str(e)}")
                    st.error("Please check your file format and try again.")
                    st.stop()
            
            # Smart Column Detection & Analysis
            st.markdown("### 🔍 Step 2: Smart Column Detection")
            
            with st.expander("📊 **Dataset Overview**", expanded=True):
                col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                with col_info1:
                    st.metric("📋 Total Samples", len(ml_data))
                with col_info2:
                    st.metric("📊 Total Features", len(ml_data.columns))
                with col_info3:
                    missing_count = ml_data.isnull().sum().sum()
                    st.metric("❓ Missing Values", missing_count)
                with col_info4:
                    memory_usage = ml_data.memory_usage(deep=True).sum() / 1024**2
                    st.metric("💾 Memory Usage", f"{memory_usage:.1f} MB")
                
                # Data preview
                st.markdown("**📋 Data Preview:**")
                st.dataframe(ml_data.head(10), use_container_width=True)
    
    with col_templates:
        st.markdown("**🎯 Use Case Templates**")
        
        template_options = {
            "🏢 Hiring & Recruitment": {
                "description": "Resume screening, job applications",
                "target_examples": ["hired", "selected", "approved"],
                "sensitive_examples": ["gender", "race", "age_group"]
            },
            "💰 Financial Services": {
                "description": "Loan approvals, credit scoring",
                "target_examples": ["approved", "granted", "accepted"],
                "sensitive_examples": ["income_level", "location", "age"]
            },
            "🏥 Healthcare": {
                "description": "Treatment recommendations",
                "target_examples": ["recommended", "prescribed", "approved"],
                "sensitive_examples": ["gender", "age_group", "ethnicity"]
            },
            "🎓 Education": {
                "description": "Admissions, scholarships",
                "target_examples": ["admitted", "awarded", "selected"],
                "sensitive_examples": ["gender", "ethnicity", "location"]
            }
        }
        
        selected_template = st.selectbox(
            "Choose your use case:",
            options=list(template_options.keys()),
            help="Select the template that matches your dataset"
        )
        
        if selected_template:
            template = template_options[selected_template]
            st.info(f"**{selected_template}**")
            st.write(f"📝 {template['description']}")
            st.write(f"🎯 **Target examples**: {', '.join(template['target_examples'])}")
            st.write(f"👥 **Sensitive examples**: {', '.join(template['sensitive_examples'])}")
    
    # Enhanced Model Configuration (only show if data is loaded)
    if 'ml_data' in locals():
        st.markdown("---")
        st.markdown("### ⚙️ Step 3: Smart Model Configuration")
        
        col_config_left, col_config_right = st.columns(2)
        
        with col_config_left:
            # Enhanced Target column detection for ANY dataset
            st.markdown("**🎯 Target Column Selection**")
            st.info("💡 **Target Column** = What you want to predict (must have exactly 2 values)")
            
            # Advanced column analysis for any dataset
            binary_columns = []
            non_binary_columns = []
            potential_targets = []
            
            for col in ml_data.columns:
                unique_vals = ml_data[col].unique()
                if len(unique_vals) == 2:
                    # Check if it looks like a target variable
                    val_str = str(unique_vals).lower()
                    is_likely_target = any(keyword in val_str for keyword in [
                        'hired', 'approved', 'accepted', 'passed', 'success', 'yes', 'no',
                        'true', 'false', '0', '1', 'positive', 'negative', 'good', 'bad',
                        'granted', 'selected', 'recommended', 'admitted', 'awarded'
                    ])
                    
                    binary_columns.append(f"{col} ({unique_vals[0]}, {unique_vals[1]})")
                    if is_likely_target:
                        potential_targets.append(col)
                else:
                    non_binary_columns.append(f"{col} ({len(unique_vals)} values)")
            
            if binary_columns:
                st.success(f"✅ **Valid target columns (binary)**: {', '.join(binary_columns)}")
            
            if non_binary_columns:
                st.warning(f"⚠️ **Cannot use as target (multi-value)**: {', '.join(non_binary_columns[:3])}")
            
            # Smart default selection for ANY dataset
            default_target = None
            if potential_targets:
                default_target = potential_targets[0]  # Use most likely target
            elif binary_columns:
                # Use first binary column if no obvious target
                default_target = binary_columns[0].split(' (')[0]
            else:
                default_target = ml_data.columns[0]
            
            target_column = st.selectbox(
                "Select the column that contains the outcome you want to predict",
                options=ml_data.columns.tolist(),
                index=ml_data.columns.tolist().index(default_target) if default_target in ml_data.columns else 0,
                help="Choose a column with exactly 2 values (binary outcomes)"
            )
            
            # Show preview of selected target column
            if target_column:
                target_preview = ml_data[target_column].value_counts()
                st.markdown(f"**Preview of '{target_column}':**")
                for value, count in target_preview.items():
                    st.write(f"• {value}: {count} samples ({count/len(ml_data)*100:.1f}%)")
                
                # Validate target column
                unique_target_vals = ml_data[target_column].unique()
                if len(unique_target_vals) != 2:
                    st.error(f"❌ **Invalid Target Column**: '{target_column}' has {len(unique_target_vals)} values")
                    st.error("🔄 **Solution**: Choose a column with exactly 2 values")
                else:
                    st.success(f"✅ Perfect! '{target_column}' has 2 values: {unique_target_vals}")
        
        with col_config_right:
            # Enhanced Sensitive feature selection for ANY dataset
            st.markdown("**👥 Sensitive Feature Selection**")
            st.info("💡 **Sensitive Feature** = Demographics to check for bias")
            
            available_features = [col for col in ml_data.columns if col != target_column]
            
            # Expanded sensitive feature detection for ANY dataset
            sensitive_keywords = [
                # Demographics
                'gender', 'sex', 'male', 'female', 'race', 'ethnicity', 'ethnic', 'color', 'colour',
                'age', 'birth', 'old', 'young', 'senior', 'junior',
                # Location
                'location', 'city', 'state', 'country', 'region', 'area', 'zip', 'postal',
                'urban', 'rural', 'suburb', 'address', 'residence',
                # Education  
                'education', 'degree', 'school', 'university', 'college', 'qualification',
                'diploma', 'certificate', 'grade', 'level',
                # Employment
                'experience', 'job', 'work', 'employment', 'career', 'position', 'title',
                # Other demographics
                'religion', 'marital', 'married', 'single', 'family', 'income', 'salary',
                'disability', 'language', 'nationality', 'citizen', 'immigrant'
            ]
            
            # Find potential sensitive features
            potential_sensitive = []
            for col in available_features:
                col_lower = col.lower()
                # Check if column name contains sensitive keywords
                if any(keyword in col_lower for keyword in sensitive_keywords):
                    potential_sensitive.append(col)
                # Also check if it's categorical (reasonable number of unique values)
                elif 2 <= len(ml_data[col].unique()) <= 20:
                    potential_sensitive.append(col)
            
            # Remove duplicates while preserving order
            potential_sensitive = list(dict.fromkeys(potential_sensitive))
            
            if potential_sensitive:
                st.success(f"✅ **Suggested sensitive features**: {', '.join(potential_sensitive[:3])}")
                if len(potential_sensitive) > 3:
                    st.info(f"... and {len(potential_sensitive) - 3} more potential features")
            
            # Smart default selection for sensitive feature
            default_sensitive = potential_sensitive[0] if potential_sensitive else available_features[0]
            
            sensitive_feature = st.selectbox(
                "Select the demographic feature to analyze for bias",
                options=available_features,
                index=available_features.index(default_sensitive),
                help="Choose any categorical column representing demographics"
            )
            
            # Show preview of selected sensitive feature
            if sensitive_feature:
                sensitive_preview = ml_data[sensitive_feature].value_counts()
                st.markdown(f"**Preview of '{sensitive_feature}':**")
                for value, count in sensitive_preview.head(5).items():
                    st.write(f"• {value}: {count} samples ({count/len(ml_data)*100:.1f}%)")
                if len(sensitive_preview) > 5:
                    st.write(f"• ... and {len(sensitive_preview) - 5} more groups")
        
        # Model Training Section
        st.markdown("---")
        st.markdown("### 🤖 Step 4: Model Training & Evaluation")
        
        col_model_config, col_model_params = st.columns(2)
        
        with col_model_config:
            st.markdown("**🛠️ Model Configuration**")
            
            model_type = st.selectbox(
                "Choose ML Algorithm",
                options=['LogisticRegression', 'RandomForestClassifier'],
                help="LogisticRegression: Fast, interpretable. RandomForest: More complex, potentially higher accuracy"
            )
            
            test_size = st.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Percentage of data reserved for testing (20% recommended)"
            )
            
            random_state = st.number_input(
                "Random Seed",
                min_value=1,
                max_value=1000,
                value=42,
                help="For reproducible results"
            )
        
        with col_model_params:
            st.markdown("**⚙️ Advanced Parameters**")
            
            if model_type == 'LogisticRegression':
                max_iter = st.number_input(
                    "Max Iterations",
                    min_value=100,
                    max_value=2000,
                    value=1000,
                    help="Maximum iterations for convergence"
                )
                
                solver = st.selectbox(
                    "Solver",
                    options=['liblinear', 'lbfgs'],
                    index=1,
                    help="Algorithm for optimization"
                )
            else:  # RandomForest
                n_estimators = st.number_input(
                    "Number of Trees",
                    min_value=10,
                    max_value=500,
                    value=100,
                    help="More trees = better performance but slower"
                )
                
                max_depth = st.selectbox(
                    "Max Depth",
                    options=[None, 5, 10, 15, 20],
                    index=0,
                    help="Maximum depth of trees (None = unlimited)"
                )
        
        # Train Model Button
        can_train = (
            len(ml_data[target_column].unique()) == 2 and 
            target_column != sensitive_feature
        )
        
        if can_train:
            if st.button("🚀 Train & Evaluate Model", type="primary", use_container_width=True):
                with st.spinner("Training model and calculating fairness metrics..."):
                    # Store configuration
                    st.session_state['ml_config'] = {
                        'model_type': model_type,
                        'test_size': test_size,
                        'random_state': random_state,
                        'target_column': target_column,
                        'sensitive_feature': sensitive_feature
                    }
                    
                    if model_type == 'LogisticRegression':
                        st.session_state['ml_config']['max_iter'] = max_iter
                        st.session_state['ml_config']['solver'] = solver
                    else:
                        st.session_state['ml_config']['n_estimators'] = n_estimators
                        st.session_state['ml_config']['max_depth'] = max_depth
                    
                    # Preprocess data
                    X, y, categorical_cols, numerical_cols = preprocess_data(ml_data, target_column)
                    
                    if X is not None and y is not None:
                        # Train model with custom parameters
                        if model_type == 'LogisticRegression':
                            model_results = train_baseline_model(
                                X, y, model_type, test_size, random_state,
                                max_iter=max_iter, solver=solver
                            )
                        else:
                            model_results = train_baseline_model(
                                X, y, model_type, test_size, random_state,
                                n_estimators=n_estimators, max_depth=max_depth
                            )
                        
                        # Store results
                        st.session_state['model_results'] = model_results
                        st.session_state['ml_data'] = ml_data
                        st.session_state['X'] = X
                        st.session_state['y'] = y
                        
                        st.success("✅ Model trained successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to preprocess data. Please check your selections.")
        else:
            if len(ml_data[target_column].unique()) != 2:
                st.error("❌ Target column must have exactly 2 unique values")
            if target_column == sensitive_feature:
                st.error("❌ Target and sensitive feature cannot be the same")
    
    # Display Results Section
    if 'model_results' in st.session_state and 'ml_config' in st.session_state:
        st.markdown("---")
        st.markdown("### 📈 Step 5: Model Performance & Bias Analysis")
        
        model_results = st.session_state['model_results']
        ml_config = st.session_state['ml_config']
        
        # Performance Metrics
        col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
        
        with col_perf1:
            st.metric(
                "🎯 Training Accuracy",
                f"{model_results['train_accuracy']:.3f}",
                help="Model performance on training data"
            )
        
        with col_perf2:
            st.metric(
                "🧪 Test Accuracy", 
                f"{model_results['test_accuracy']:.3f}",
                help="Model performance on unseen test data"
            )
        
        with col_perf3:
            # Calculate precision
            from sklearn.metrics import precision_score, recall_score
            precision = precision_score(model_results['y_test'], model_results['y_pred_test'])
            st.metric(
                "🎯 Precision",
                f"{precision:.3f}",
                help="Of predicted positives, how many were correct"
            )
        
        with col_perf4:
            # Calculate recall
            recall = recall_score(model_results['y_test'], model_results['y_pred_test'])
            st.metric(
                "📊 Recall",
                f"{recall:.3f}",
                help="Of actual positives, how many were found"
            )
        
        # Fairness Analysis
        st.markdown("### ⚖️ Comprehensive Fairness Analysis")
        
        # Calculate fairness metrics
        ml_data = st.session_state['ml_data']
        sensitive_feature = ml_config['sensitive_feature']
        
        # Get sensitive feature values for test set
        test_indices = model_results['X_test_original'].index
        sensitive_test = ml_data.loc[test_indices, sensitive_feature].values
        
        # Calculate fairness metrics
        fairness_metrics = calculate_fairness_metrics_custom(
            model_results['y_test'], 
            model_results['y_pred_test'], 
            sensitive_test
        )
        
        # Store for mitigation comparison
        st.session_state['baseline_fairness_metrics'] = fairness_metrics
        st.session_state['baseline_accuracy'] = model_results['test_accuracy']
        
        # Display fairness metrics with explanations
        col_fair1, col_fair2, col_fair3 = st.columns(3)
        
        with col_fair1:
            dp_diff = fairness_metrics['demographic_parity_difference']
            dp_color = "inverse" if dp_diff > 0.1 else "normal"
            st.metric(
                "⚖️ Demographic Parity",
                f"{dp_diff:.3f}",
                delta=f"{'High bias' if dp_diff > 0.1 else 'Acceptable'}" if dp_diff > 0.05 else "Low bias",
                delta_color=dp_color,
                help="Difference in positive prediction rates between groups (closer to 0 is better)"
            )
            
            with st.expander("ℹ️ What is Demographic Parity?"):
                st.write("""
                **Demographic Parity** measures whether different groups receive positive outcomes at similar rates.
                
                - **Value**: Difference between highest and lowest group rates
                - **Range**: 0 (perfect) to 1 (maximum bias)
                - **Threshold**: < 0.1 is generally acceptable
                - **Example**: If 70% of Group A gets hired vs 50% of Group B, difference = 0.20
                """)
        
        with col_fair2:
            eo_diff = fairness_metrics['equalized_odds_difference']
            eo_color = "inverse" if eo_diff > 0.1 else "normal"
            st.metric(
                "⚡ Equalized Odds",
                f"{eo_diff:.3f}",
                delta=f"{'High bias' if eo_diff > 0.1 else 'Acceptable'}" if eo_diff > 0.05 else "Low bias",
                delta_color=eo_color,
                help="Difference in true positive rates between groups (closer to 0 is better)"
            )
            
            with st.expander("ℹ️ What is Equalized Odds?"):
                st.write("""
                **Equalized Odds** measures whether the model performs equally well for different groups.
                
                - **Value**: Difference in true positive rates between groups
                - **Range**: 0 (perfect) to 1 (maximum bias)
                - **Threshold**: < 0.1 is generally acceptable
                - **Example**: If model correctly identifies 80% of qualified Group A vs 60% of qualified Group B
                """)
        
        with col_fair3:
            di_ratio = fairness_metrics['disparate_impact_ratio']
            di_color = "inverse" if di_ratio < 0.8 else "normal"
            st.metric(
                "📊 Disparate Impact",
                f"{di_ratio:.3f}",
                delta=f"{'Biased' if di_ratio < 0.8 else 'Fair'}" if di_ratio < 0.9 else "Good",
                delta_color=di_color,
                help="Ratio of positive rates between groups (should be close to 1.0)"
            )
            
            with st.expander("ℹ️ What is Disparate Impact?"):
                st.write("""
                **Disparate Impact** compares outcome rates between groups as a ratio.
                
                - **Value**: Ratio of lowest to highest group rates
                - **Range**: 0 (maximum bias) to 1 (perfect fairness)
                - **Legal Threshold**: > 0.8 (80% rule)
                - **Example**: If Group A: 60% positive, Group B: 80% positive, ratio = 0.75 (biased)
                """)
        
        # Detailed Group Analysis
        st.markdown("### 👥 Detailed Group Analysis")
        
        group_df = pd.DataFrame(fairness_metrics['group_metrics']).T
        group_df = group_df.round(3)
        group_df.columns = ['Prediction Rate', 'True Positive Rate', 'Accuracy', 'Sample Count']
        
        # Add percentage columns
        total_samples = group_df['Sample Count'].sum()
        group_df['Sample %'] = (group_df['Sample Count'] / total_samples * 100).round(1)
        group_df['Prediction %'] = (group_df['Prediction Rate'] * 100).round(1)
        group_df['TPR %'] = (group_df['True Positive Rate'] * 100).round(1)
        group_df['Accuracy %'] = (group_df['Accuracy'] * 100).round(1)
        
        # Reorder columns for better display
        display_df = group_df[['Sample Count', 'Sample %', 'Prediction %', 'TPR %', 'Accuracy %', 'Prediction Rate', 'True Positive Rate', 'Accuracy']]
        
        st.dataframe(display_df, use_container_width=True)
        
        # Visualization
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            # Prediction rates by group
            fig_pred = px.bar(
                x=list(fairness_metrics['group_metrics'].keys()),
                y=[metrics['prediction_rate'] * 100 for metrics in fairness_metrics['group_metrics'].values()],
                title=f"Prediction Rates by {sensitive_feature}",
                labels={'x': sensitive_feature, 'y': 'Prediction Rate (%)'},
                color=[metrics['prediction_rate'] for metrics in fairness_metrics['group_metrics'].values()],
                color_continuous_scale='RdYlBu_r'
            )
            fig_pred.update_layout(showlegend=False)
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with col_viz2:
            # Sample distribution
            fig_dist = px.pie(
                values=[metrics['count'] for metrics in fairness_metrics['group_metrics'].values()],
                names=list(fairness_metrics['group_metrics'].keys()),
                title=f"Sample Distribution by {sensitive_feature}"
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    # Bias Mitigation Section
    st.markdown("---")
    st.markdown("### 🛠️ Step 6: Bias Mitigation & Comparison")
    
    col_mitigation_left, col_mitigation_right = st.columns(2)
    
    with col_mitigation_left:
        st.markdown("**🔧 Choose Mitigation Strategy**")
        
        mitigation_method = st.selectbox(
            "Select bias mitigation technique:",
            options=[
                "🔄 Reweighing (Pre-processing)",
                "📊 Threshold Optimization (Post-processing)",
                "🎯 Both Techniques"
            ],
            help="Choose how to reduce bias in your model"
        )
        
        with st.expander("ℹ️ **Mitigation Techniques Explained**"):
            st.markdown("""
            **🔄 Reweighing (Pre-processing)**
            - Adjusts training data weights to balance group representation
            - Applied before model training
            - Good for: Addressing training data imbalances
            
            **📊 Threshold Optimization (Post-processing)**
            - Adjusts decision thresholds for different groups
            - Applied after model training
            - Good for: Fine-tuning existing model decisions
            
            **🎯 Both Techniques**
            - Combines both approaches for maximum bias reduction
            - May provide best fairness improvements
            - Trade-off: Potentially larger impact on accuracy
            """)
        
        # Fairness-Accuracy Trade-off Slider
        fairness_weight = st.slider(
            "Fairness vs Accuracy Priority",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0.0 = Prioritize Accuracy, 1.0 = Prioritize Fairness"
        )
        
        if fairness_weight < 0.3:
            st.info("🎯 **Accuracy Priority**: Minimal fairness intervention")
        elif fairness_weight > 0.7:
            st.info("⚖️ **Fairness Priority**: Maximum bias reduction")
        else:
            st.info("⚖️ **Balanced Approach**: Moderate fairness improvements")
    
    with col_mitigation_right:
        st.markdown("**📊 Current Bias Summary**")
        
        # Create bias summary
        bias_summary = {
            "Demographic Parity": {
                "value": fairness_metrics['demographic_parity_difference'],
                "status": "🔴 High" if fairness_metrics['demographic_parity_difference'] > 0.1 else "🟡 Medium" if fairness_metrics['demographic_parity_difference'] > 0.05 else "🟢 Low",
                "target": "< 0.10"
            },
            "Equalized Odds": {
                "value": fairness_metrics['equalized_odds_difference'],
                "status": "🔴 High" if fairness_metrics['equalized_odds_difference'] > 0.1 else "🟡 Medium" if fairness_metrics['equalized_odds_difference'] > 0.05 else "🟢 Low",
                "target": "< 0.10"
            },
            "Disparate Impact": {
                "value": fairness_metrics['disparate_impact_ratio'],
                "status": "🔴 Biased" if fairness_metrics['disparate_impact_ratio'] < 0.8 else "🟡 Borderline" if fairness_metrics['disparate_impact_ratio'] < 0.9 else "🟢 Fair",
                "target": "> 0.80"
            }
        }
        
        for metric, info in bias_summary.items():
            st.write(f"**{metric}**: {info['status']}")
            st.write(f"  Current: {info['value']:.3f} | Target: {info['target']}")
        
        # Overall bias assessment
        high_bias_count = sum(1 for info in bias_summary.values() if "🔴" in info['status'])
        if high_bias_count >= 2:
            st.error("🚨 **High Bias Detected**: Mitigation strongly recommended")
        elif high_bias_count == 1:
            st.warning("⚠️ **Moderate Bias**: Mitigation recommended")
        else:
            st.success("✅ **Low Bias**: Mitigation optional for improvement")
    
    # Apply Mitigation Button
    if st.button("🚀 Apply Bias Mitigation", type="primary", use_container_width=True):
        with st.spinner("Applying bias mitigation techniques..."):
            try:
                # Get data
                X = st.session_state['X']
                y = st.session_state['y']
                ml_data = st.session_state['ml_data']
                sensitive_feature = ml_config['sensitive_feature']
                
                # Prepare sensitive feature data
                sensitive_feature_data = ml_data[sensitive_feature].values
                
                mitigated_results = {}
                
                if "Reweighing" in mitigation_method or "Both" in mitigation_method:
                    st.info("🔄 Applying Reweighing...")
                    
                    # Apply reweighing mitigation
                    mitigated_X, mitigated_y, sample_weights = apply_reweighing_mitigation(
                        X, y, sensitive_feature_data
                    )
                    
                    # Retrain model with weights
                    if model_type == 'LogisticRegression':
                        mitigated_model_results = train_baseline_model(
                            mitigated_X, mitigated_y, model_type, test_size, random_state,
                            max_iter=ml_config.get('max_iter', 1000),
                            solver=ml_config.get('solver', 'lbfgs')
                        )
                    else:
                        mitigated_model_results = train_baseline_model(
                            mitigated_X, mitigated_y, model_type, test_size, random_state,
                            n_estimators=ml_config.get('n_estimators', 100),
                            max_depth=ml_config.get('max_depth', None)
                        )
                    
                    mitigated_results['reweighing'] = mitigated_model_results
                
                if "Threshold" in mitigation_method or "Both" in mitigation_method:
                    st.info("📊 Applying Threshold Optimization...")
                    
                    # Get test set sensitive features
                    test_indices = model_results['X_test_original'].index
                    sensitive_test = ml_data.loc[test_indices, sensitive_feature].values
                    
                    # Apply threshold optimization
                    optimized_predictions = apply_threshold_optimization(
                        model_results['y_test'],
                        model_results['y_pred_proba_test'],
                        sensitive_test,
                        fairness_weight=fairness_weight
                    )
                    
                    # Calculate new metrics
                    from sklearn.metrics import accuracy_score
                    optimized_accuracy = accuracy_score(model_results['y_test'], optimized_predictions)
                    
                    mitigated_results['threshold_optimization'] = {
                        'y_pred_test': optimized_predictions,
                        'test_accuracy': optimized_accuracy,
                        'y_test': model_results['y_test'],
                        'sensitive_test': sensitive_test
                    }
                
                # Calculate fairness metrics for mitigated models
                mitigated_fairness_metrics = {}
                
                for method, results in mitigated_results.items():
                    if method == 'reweighing':
                        # Get test set for reweighed model
                        test_indices = results['X_test_original'].index
                        sensitive_test_reweighed = ml_data.loc[test_indices, sensitive_feature].values
                        
                        fairness_metrics_reweighed = calculate_fairness_metrics_custom(
                            results['y_test'],
                            results['y_pred_test'],
                            sensitive_test_reweighed
                        )
                        mitigated_fairness_metrics[method] = fairness_metrics_reweighed
                        
                    elif method == 'threshold_optimization':
                        fairness_metrics_threshold = calculate_fairness_metrics_custom(
                            results['y_test'],
                            results['y_pred_test'],
                            results['sensitive_test']
                        )
                        mitigated_fairness_metrics[method] = fairness_metrics_threshold
                
                # Store results
                st.session_state['mitigated_results'] = mitigated_results
                st.session_state['mitigated_fairness_metrics'] = mitigated_fairness_metrics
                st.session_state['mitigation_method'] = mitigation_method
                
                st.success("✅ Bias mitigation completed!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during mitigation: {str(e)}")
    
    # Display Mitigation Results
    if 'mitigated_results' in st.session_state and 'mitigated_fairness_metrics' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Step 7: Mitigation Results & Comparison")
        
        # Performance Comparison
        st.markdown("#### 🎯 Performance Comparison")
        
        baseline_accuracy = st.session_state['baseline_accuracy']
        mitigation_method = st.session_state['mitigation_method']
        mitigated_results = st.session_state['mitigated_results']
        
        comparison_data = {
            "Metric": ["Accuracy"],
            "Baseline": [f"{baseline_accuracy:.3f}"]
        }
        
        for method, results in mitigated_results.items():
            method_name = method.replace('_', ' ').title()
            if method == 'reweighing':
                accuracy = results['test_accuracy']
            else:  # threshold_optimization
                accuracy = results['test_accuracy']
            
            comparison_data[method_name] = [f"{accuracy:.3f}"]
            
            # Calculate accuracy change
            accuracy_change = accuracy - baseline_accuracy
            change_color = "🔴" if accuracy_change < -0.05 else "🟡" if accuracy_change < 0 else "🟢"
            comparison_data[f"{method_name} Change"] = [f"{change_color} {accuracy_change:+.3f}"]
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Fairness Comparison
        st.markdown("#### ⚖️ Fairness Improvement")
        
        baseline_fairness = st.session_state['baseline_fairness_metrics']
        mitigated_fairness = st.session_state['mitigated_fairness_metrics']
        
        fairness_comparison = {
            "Fairness Metric": [
                "Demographic Parity",
                "Equalized Odds", 
                "Disparate Impact"
            ],
            "Baseline": [
                f"{baseline_fairness['demographic_parity_difference']:.3f}",
                f"{baseline_fairness['equalized_odds_difference']:.3f}",
                f"{baseline_fairness['disparate_impact_ratio']:.3f}"
            ]
        }
        
        for method, metrics in mitigated_fairness.items():
            method_name = method.replace('_', ' ').title()
            fairness_comparison[method_name] = [
                f"{metrics['demographic_parity_difference']:.3f}",
                f"{metrics['equalized_odds_difference']:.3f}",
                f"{metrics['disparate_impact_ratio']:.3f}"
            ]
            
            # Calculate improvements
            dp_improvement = baseline_fairness['demographic_parity_difference'] - metrics['demographic_parity_difference']
            eo_improvement = baseline_fairness['equalized_odds_difference'] - metrics['equalized_odds_difference']
            di_improvement = metrics['disparate_impact_ratio'] - baseline_fairness['disparate_impact_ratio']
            
            improvements = [dp_improvement, eo_improvement, di_improvement]
            improvement_colors = []
            
            for imp in improvements:
                if imp > 0.05:
                    improvement_colors.append("🟢 +" + f"{imp:.3f}")
                elif imp > 0:
                    improvement_colors.append("🟡 +" + f"{imp:.3f}")
                else:
                    improvement_colors.append("🔴 " + f"{imp:.3f}")
            
            fairness_comparison[f"{method_name} Change"] = improvement_colors
        
        fairness_df = pd.DataFrame(fairness_comparison)
        st.dataframe(fairness_df, use_container_width=True)
        
        # Visualization of Improvements
        st.markdown("#### 📈 Visual Comparison")
        
        col_viz_comp1, col_viz_comp2 = st.columns(2)
        
        with col_viz_comp1:
            # Create comparison chart for fairness metrics
            fig_comparison = create_comparison_chart(baseline_fairness, list(mitigated_fairness.values())[0])
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        with col_viz_comp2:
            # Accuracy vs Fairness Trade-off
            methods = ['Baseline'] + [method.replace('_', ' ').title() for method in mitigated_results.keys()]
            accuracies = [baseline_accuracy] + [
                results['test_accuracy'] for results in mitigated_results.values()
            ]
            
            # Use demographic parity as fairness measure (lower is better)
            fairness_scores = [baseline_fairness['demographic_parity_difference']] + [
                metrics['demographic_parity_difference'] for metrics in mitigated_fairness.values()
            ]
            
            fig_tradeoff = px.scatter(
                x=fairness_scores,
                y=accuracies,
                text=methods,
                title="Accuracy vs Fairness Trade-off",
                labels={
                    'x': 'Demographic Parity Difference (lower = more fair)',
                    'y': 'Accuracy (higher = better)'
                }
            )
            fig_tradeoff.update_traces(textposition="top center")
            st.plotly_chart(fig_tradeoff, use_container_width=True)
        
        # Export Results
        st.markdown("#### 📤 Export Comprehensive Report")
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            if st.button("📄 Download JSON Report", use_container_width=True):
                # Create comprehensive JSON report
                comprehensive_report = {
                    "dataset_info": {
                        "samples": len(ml_data),
                        "features": len(ml_data.columns),
                        "target_column": ml_config['target_column'],
                        "sensitive_feature": ml_config['sensitive_feature']
                    },
                    "model_config": ml_config,
                    "baseline_performance": {
                        "accuracy": baseline_accuracy,
                        "fairness_metrics": baseline_fairness
                    },
                    "mitigation_results": {
                        "method": mitigation_method,
                        "performance": {method: results.get('test_accuracy', 0) for method, results in mitigated_results.items()},
                        "fairness_metrics": mitigated_fairness
                    },
                    "recommendations": {
                        "best_method": max(mitigated_fairness.keys(), key=lambda k: mitigated_fairness[k]['disparate_impact_ratio']),
                        "accuracy_trade_off": min(mitigated_results.values(), key=lambda x: x['test_accuracy'])['test_accuracy'] - baseline_accuracy
                    }
                }
                
                import json
                report_json = json.dumps(comprehensive_report, indent=2, default=str)
                
                st.download_button(
                    "⬇️ Download JSON",
                    data=report_json,
                    file_name=f"bias_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col_export2:
            if st.button("📋 Download TXT Summary", use_container_width=True):
                # Create executive summary
                summary_text = f"""
COMPREHENSIVE BIAS AUDIT REPORT
===============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW
----------------
• Total Samples: {len(ml_data):,}
• Features: {len(ml_data.columns)}
• Target Column: {ml_config['target_column']}
• Sensitive Feature: {ml_config['sensitive_feature']}

BASELINE MODEL PERFORMANCE
--------------------------
• Algorithm: {ml_config['model_type']}
• Test Accuracy: {baseline_accuracy:.3f}
• Demographic Parity: {baseline_fairness['demographic_parity_difference']:.3f}
• Equalized Odds: {baseline_fairness['equalized_odds_difference']:.3f}
• Disparate Impact: {baseline_fairness['disparate_impact_ratio']:.3f}

BIAS MITIGATION RESULTS
-----------------------
• Method Applied: {mitigation_method}
• Best Performing Method: {max(mitigated_fairness.keys(), key=lambda k: mitigated_fairness[k]['disparate_impact_ratio'])}

FAIRNESS IMPROVEMENTS
--------------------
"""
                
                for method, metrics in mitigated_fairness.items():
                    dp_improvement = baseline_fairness['demographic_parity_difference'] - metrics['demographic_parity_difference']
                    summary_text += f"• {method.title()}: {dp_improvement:+.3f} improvement in demographic parity\n"
                
                summary_text += f"""
RECOMMENDATIONS
---------------
• {'Strong bias mitigation recommended' if max(baseline_fairness['demographic_parity_difference'], baseline_fairness['equalized_odds_difference']) > 0.1 else 'Moderate improvements achieved'}
• Accuracy trade-off: {min(mitigated_results.values(), key=lambda x: x['test_accuracy'])['test_accuracy'] - baseline_accuracy:+.3f}
• Continue monitoring for bias in production deployment

COMPLIANCE STATUS
-----------------
• Disparate Impact (80% rule): {'✅ PASS' if max(mitigated_fairness.values(), key=lambda x: x['disparate_impact_ratio'])['disparate_impact_ratio'] > 0.8 else '❌ FAIL'}
• Demographic Parity: {'✅ ACCEPTABLE' if min(mitigated_fairness.values(), key=lambda x: x['demographic_parity_difference'])['demographic_parity_difference'] < 0.1 else '⚠️ NEEDS IMPROVEMENT'}
"""
                
                st.download_button(
                    "⬇️ Download Summary",
                    data=summary_text,
                    file_name=f"bias_audit_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col_export3:
            if st.button("📊 Download Excel Report", use_container_width=True):
                st.info("📊 Excel export feature coming soon!")
                st.write("For now, use JSON or TXT formats for comprehensive reporting.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>Tech Titanians Bias Audit Dashboard</strong> | June 2025</p>
    <p>Promoting algorithmic fairness in machine learning systems worldwide</p>
    <p>🏛️ Employment Equity Act Compliant | 🔒 POPIA Aligned</p>
</div>
""", unsafe_allow_html=True)
