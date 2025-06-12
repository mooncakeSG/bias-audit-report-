// Global variables
let uploadedData = null;
let auditResults = null;
let biasMetrics = null;
let charts = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeFileUpload();
    setupDragAndDrop();
    loadExistingResults();
});

// File upload initialization
function initializeFileUpload() {
    const fileInput = document.getElementById('fileInput');
    fileInput.addEventListener('change', handleFileUpload);
}

// Setup drag and drop functionality
function setupDragAndDrop() {
    const uploadArea = document.getElementById('uploadArea');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    uploadArea.addEventListener('drop', handleDrop, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    document.getElementById('uploadArea').classList.add('drag-over');
}

function unhighlight(e) {
    document.getElementById('uploadArea').classList.remove('drag-over');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        handleFileUpload({ target: { files: files } });
    }
}

// Handle file upload
function handleFileUpload(event) {
    const file = event.target.files[0];
    
    if (!file) return;
    
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showToast('Please upload a CSV file.', 'error');
        return;
    }
    
    // Clear previous results when uploading new file
    document.getElementById('resultsSection').style.display = 'none';
    
    showToast('Processing file...', 'info');
    
    Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        transformHeader: function(header) {
            // Clean up header names
            return header.trim();
        },
        complete: function(results) {
            if (results.errors && results.errors.length > 0) {
                console.warn('CSV parsing warnings:', results.errors);
                // Only show error if there are critical errors
                const criticalErrors = results.errors.filter(error => 
                    error.type === 'Delimiter' || error.type === 'Quotes'
                );
                if (criticalErrors.length > 0) {
                    showToast('Error parsing CSV file. Please check the format.', 'error');
                    console.error('Critical CSV parsing errors:', criticalErrors);
                    return;
                }
            }
            
            if (!results.data || results.data.length === 0) {
                showToast('CSV file appears to be empty.', 'error');
                return;
            }
            
            // Filter out completely empty rows
            uploadedData = results.data.filter(row => 
                Object.values(row).some(val => val !== null && val !== undefined && val !== '')
            );
            
            if (uploadedData.length === 0) {
                showToast('No valid data found in CSV file.', 'error');
                return;
            }
            
            setupColumnConfiguration();
            showToast(`File uploaded successfully! ${uploadedData.length} records loaded.`, 'success');
        },
        error: function(error) {
            showToast('Error reading file. Please try again.', 'error');
            console.error('File reading error:', error);
        }
    });
}

// Setup column configuration
function setupColumnConfiguration() {
    if (!uploadedData || uploadedData.length === 0) return;
    
    const columns = Object.keys(uploadedData[0]);
    
    // Populate target column dropdown
    populateSelect('targetColumn', columns);
    
    // Populate prediction column dropdown
    populateSelect('predictionColumn', columns);
    
    // Create checkboxes for protected attributes
    createProtectedAttributeCheckboxes(columns);
    
    // Show configuration section
    document.getElementById('configSection').style.display = 'block';
    document.getElementById('configSection').classList.add('fade-in');
}

function populateSelect(selectId, options) {
    const select = document.getElementById(selectId);
    select.innerHTML = '<option value="">Select column...</option>';
    
    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option;
        select.appendChild(optionElement);
    });
}

function createProtectedAttributeCheckboxes(columns) {
    const container = document.getElementById('protectedAttributesContainer');
    container.innerHTML = '';
    
    columns.forEach(column => {
        const checkboxItem = document.createElement('div');
        checkboxItem.className = 'checkbox-item';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `attr_${column}`;
        checkbox.value = column;
        
        const label = document.createElement('label');
        label.htmlFor = `attr_${column}`;
        label.textContent = column;
        
        checkboxItem.appendChild(checkbox);
        checkboxItem.appendChild(label);
        container.appendChild(checkboxItem);
    });
}

// Run bias audit
async function runBiasAudit() {
    if (!validateConfiguration()) return;
    
    const config = getConfiguration();
    showLoading(true);
    
    try {
        // Calculate fairness metrics
        auditResults = calculateFairnessMetrics(uploadedData, config);
        
        // Calculate bias metrics
        biasMetrics = calculateBiasMetrics(auditResults, config.protectedAttributes);
        
        // Display results
        displayResults();
        
        showToast('Bias audit completed successfully!', 'success');
    } catch (error) {
        console.error('Error running bias audit:', error);
        showToast('Error running bias audit. Please check your data and configuration.', 'error');
    } finally {
        showLoading(false);
    }
}

function validateConfiguration() {
    const targetColumn = document.getElementById('targetColumn').value;
    const predictionColumn = document.getElementById('predictionColumn').value;
    const protectedAttributes = getSelectedProtectedAttributes();
    
    if (!targetColumn) {
        showToast('Please select a target column.', 'warning');
        return false;
    }
    
    if (!predictionColumn) {
        showToast('Please select a prediction column.', 'warning');
        return false;
    }
    
    if (protectedAttributes.length === 0) {
        showToast('Please select at least one protected attribute.', 'warning');
        return false;
    }
    
    return true;
}

function getConfiguration() {
    return {
        targetColumn: document.getElementById('targetColumn').value,
        predictionColumn: document.getElementById('predictionColumn').value,
        protectedAttributes: getSelectedProtectedAttributes()
    };
}

function getSelectedProtectedAttributes() {
    const checkboxes = document.querySelectorAll('#protectedAttributesContainer input[type="checkbox"]:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

// Calculate fairness metrics
function calculateFairnessMetrics(data, config) {
    const results = [];
    
    config.protectedAttributes.forEach(attr => {
        const groups = [...new Set(data.map(row => row[attr]))].filter(group => group !== '');
        
        groups.forEach(group => {
            const groupData = data.filter(row => row[attr] === group);
            
            if (groupData.length === 0) return;
            
            const yTrue = groupData.map(row => parseInt(row[config.targetColumn]) || 0);
            const yPred = groupData.map(row => parseInt(row[config.predictionColumn]) || 0);
            
            const metrics = calculateMetrics(yTrue, yPred);
            
            results.push({
                protectedAttribute: attr,
                group: group,
                sampleSize: groupData.length,
                ...metrics
            });
        });
    });
    
    return results;
}

function calculateMetrics(yTrue, yPred) {
    const correct = yTrue.reduce((sum, actual, i) => sum + (actual === yPred[i] ? 1 : 0), 0);
    const accuracy = correct / yTrue.length;
    
    // Calculate confusion matrix elements
    let tp = 0, tn = 0, fp = 0, fn = 0;
    
    for (let i = 0; i < yTrue.length; i++) {
        if (yTrue[i] === 1 && yPred[i] === 1) tp++;
        else if (yTrue[i] === 0 && yPred[i] === 0) tn++;
        else if (yTrue[i] === 0 && yPred[i] === 1) fp++;
        else if (yTrue[i] === 1 && yPred[i] === 0) fn++;
    }
    
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1Score = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    
    const positivePredictionRate = yPred.filter(pred => pred === 1).length / yPred.length;
    const falsePositiveRate = fp + tn > 0 ? fp / (fp + tn) : 0;
    
    return {
        accuracy,
        precision,
        recall,
        f1Score,
        positivePredictionRate,
        truePositiveRate: recall,
        falsePositiveRate
    };
}

// Calculate bias metrics
function calculateBiasMetrics(results, protectedAttributes) {
    const biasResults = [];
    
    protectedAttributes.forEach(attr => {
        const attrData = results.filter(result => result.protectedAttribute === attr);
        
        if (attrData.length < 2) return;
        
        const accuracies = attrData.map(d => d.accuracy);
        const posRates = attrData.map(d => d.positivePredictionRate);
        const tprRates = attrData.map(d => d.truePositiveRate);
        const fprRates = attrData.map(d => d.falsePositiveRate);
        
        const demographicParityDiff = Math.max(...posRates) - Math.min(...posRates);
        const accuracyDiff = Math.max(...accuracies) - Math.min(...accuracies);
        const tprDiff = Math.max(...tprRates) - Math.min(...tprRates);
        const fprDiff = Math.max(...fprRates) - Math.min(...fprRates);
        const equalizedOddsDiff = Math.max(tprDiff, fprDiff);
        
        const biasSeverity = assessBiasSeverity(demographicParityDiff, equalizedOddsDiff);
        
        biasResults.push({
            protectedAttribute: attr,
            demographicParityDifference: demographicParityDiff,
            equalizedOddsDifference: equalizedOddsDiff,
            accuracyDifference: accuracyDiff,
            biasSeverity
        });
    });
    
    return biasResults;
}

function assessBiasSeverity(dpDiff, eoDiff) {
    const maxDiff = Math.max(dpDiff, eoDiff);
    
    if (maxDiff < 0.05) return 'Low';
    else if (maxDiff < 0.1) return 'Moderate';
    else if (maxDiff < 0.2) return 'High';
    else return 'Severe';
}

// Display results
function displayResults() {
    // Update summary cards
    updateSummaryCards();
    
    // Create charts
    createCharts();
    
    // Populate results table
    populateResultsTable();
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsSection').classList.add('fade-in');
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

function updateSummaryCards() {
    const overallSeverity = getOverallSeverity();
    const attributesCount = biasMetrics.length;
    const maxBiasScore = getMaxBiasScore();
    const totalSamples = auditResults.reduce((sum, result) => sum + result.sampleSize, 0);
    
    document.getElementById('overallSeverity').textContent = overallSeverity;
    document.getElementById('attributesCount').textContent = attributesCount;
    document.getElementById('maxBiasScore').textContent = (maxBiasScore * 100).toFixed(1) + '%';
    document.getElementById('totalSamples').textContent = totalSamples.toLocaleString();
}

function getOverallSeverity() {
    if (!biasMetrics || biasMetrics.length === 0) return 'N/A';
    
    const severities = biasMetrics.map(metric => metric.biasSeverity);
    
    if (severities.includes('Severe')) return 'Severe';
    if (severities.includes('High')) return 'High';
    if (severities.includes('Moderate')) return 'Moderate';
    return 'Low';
}

function getMaxBiasScore() {
    if (!biasMetrics || biasMetrics.length === 0) return 0;
    
    const allDiffs = biasMetrics.flatMap(metric => [
        metric.demographicParityDifference,
        metric.equalizedOddsDifference,
        metric.accuracyDifference
    ]);
    
    return Math.max(...allDiffs);
}

// Create charts
function createCharts() {
    createAccuracyChart();
    createSeverityChart();
    createFairnessChart();
    createDemographicChart();
}

function createAccuracyChart() {
    const ctx = document.getElementById('accuracyChart').getContext('2d');
    
    const data = {
        labels: auditResults.map(result => `${result.protectedAttribute}: ${result.group}`),
        datasets: [{
            label: 'Accuracy',
            data: auditResults.map(result => result.accuracy),
            backgroundColor: 'rgba(102, 126, 234, 0.6)',
            borderColor: 'rgba(102, 126, 234, 1)',
            borderWidth: 1
        }]
    };
    
    const config = {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    };
    
    if (charts.accuracy) charts.accuracy.destroy();
    charts.accuracy = new Chart(ctx, config);
}

function createSeverityChart() {
    const ctx = document.getElementById('severityChart').getContext('2d');
    
    const severityCounts = biasMetrics.reduce((counts, metric) => {
        counts[metric.biasSeverity] = (counts[metric.biasSeverity] || 0) + 1;
        return counts;
    }, {});
    
    const data = {
        labels: Object.keys(severityCounts),
        datasets: [{
            data: Object.values(severityCounts),
            backgroundColor: [
                '#4CAF50', // Low - Green
                '#FF9800', // Moderate - Orange  
                '#F44336', // High - Red
                '#9C27B0'  // Severe - Purple
            ]
        }]
    };
    
    const config = {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    };
    
    if (charts.severity) charts.severity.destroy();
    charts.severity = new Chart(ctx, config);
}

function createFairnessChart() {
    const ctx = document.getElementById('fairnessChart').getContext('2d');
    
    const attributes = [...new Set(auditResults.map(result => result.protectedAttribute))];
    
    const datasets = [
        {
            label: 'Precision',
            data: attributes.map(attr => {
                const attrResults = auditResults.filter(result => result.protectedAttribute === attr);
                return attrResults.reduce((sum, result) => sum + result.precision, 0) / attrResults.length;
            }),
            backgroundColor: 'rgba(255, 99, 132, 0.6)'
        },
        {
            label: 'Recall',
            data: attributes.map(attr => {
                const attrResults = auditResults.filter(result => result.protectedAttribute === attr);
                return attrResults.reduce((sum, result) => sum + result.recall, 0) / attrResults.length;
            }),
            backgroundColor: 'rgba(54, 162, 235, 0.6)'
        },
        {
            label: 'F1 Score',
            data: attributes.map(attr => {
                const attrResults = auditResults.filter(result => result.protectedAttribute === attr);
                return attrResults.reduce((sum, result) => sum + result.f1Score, 0) / attrResults.length;
            }),
            backgroundColor: 'rgba(255, 206, 86, 0.6)'
        }
    ];
    
    const config = {
        type: 'radar',
        data: {
            labels: attributes,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    };
    
    if (charts.fairness) charts.fairness.destroy();
    charts.fairness = new Chart(ctx, config);
}

function createDemographicChart() {
    const ctx = document.getElementById('demographicChart').getContext('2d');
    
    const data = {
        labels: auditResults.map(result => `${result.protectedAttribute}: ${result.group}`),
        datasets: [{
            label: 'Positive Prediction Rate',
            data: auditResults.map(result => result.positivePredictionRate),
            backgroundColor: 'rgba(75, 192, 192, 0.6)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
        }]
    };
    
    const config = {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    };
    
    if (charts.demographic) charts.demographic.destroy();
    charts.demographic = new Chart(ctx, config);
}

// Populate results table
function populateResultsTable() {
    const tbody = document.getElementById('resultsTableBody');
    tbody.innerHTML = '';
    
    auditResults.forEach(result => {
        const row = document.createElement('tr');
        
        const biasStatus = getBiasStatusForGroup(result);
        
        row.innerHTML = `
            <td>${result.protectedAttribute}</td>
            <td>${result.group}</td>
            <td>${result.sampleSize.toLocaleString()}</td>
            <td>${(result.accuracy * 100).toFixed(1)}%</td>
            <td>${(result.precision * 100).toFixed(1)}%</td>
            <td>${(result.recall * 100).toFixed(1)}%</td>
            <td>${(result.f1Score * 100).toFixed(1)}%</td>
            <td>${(result.positivePredictionRate * 100).toFixed(1)}%</td>
            <td><span class="bias-status bias-${biasStatus.toLowerCase()}">${biasStatus}</span></td>
        `;
        
        tbody.appendChild(row);
    });
}

function getBiasStatusForGroup(result) {
    const attrBias = biasMetrics.find(bias => bias.protectedAttribute === result.protectedAttribute);
    return attrBias ? attrBias.biasSeverity : 'Unknown';
}

// Export functions
function exportToCSV() {
    if (!auditResults) {
        showToast('No results to export.', 'warning');
        return;
    }
    
    const csvContent = convertToCSV(auditResults);
    downloadFile(csvContent, 'bias_audit_results.csv', 'text/csv');
    showToast('Results exported to CSV.', 'success');
}

function exportReport() {
    if (!auditResults || !biasMetrics) {
        showToast('No results to export.', 'warning');
        return;
    }
    
    const report = generateTextReport();
    downloadFile(report, 'bias_audit_report.txt', 'text/plain');
    showToast('Report exported successfully.', 'success');
}

function exportCharts() {
    showToast('Chart export feature coming soon.', 'info');
}

function convertToCSV(data) {
    if (!data || data.length === 0) return '';
    
    const headers = Object.keys(data[0]);
    const csvRows = [headers.join(',')];
    
    data.forEach(row => {
        const values = headers.map(header => {
            const value = row[header];
            return typeof value === 'string' ? `"${value}"` : value;
        });
        csvRows.push(values.join(','));
    });
    
    return csvRows.join('\n');
}

function generateTextReport() {
    const overallSeverity = getOverallSeverity();
    const maxBiasScore = getMaxBiasScore();
    const totalSamples = auditResults.reduce((sum, result) => sum + result.sampleSize, 0);
    
    let report = `BIAS AUDIT REPORT\n`;
    report += `=================\n\n`;
    report += `Generated on: ${new Date().toLocaleString()}\n\n`;
    
    report += `SUMMARY\n`;
    report += `-------\n`;
    report += `Overall Bias Severity: ${overallSeverity}\n`;
    report += `Attributes Analyzed: ${biasMetrics.length}\n`;
    report += `Maximum Bias Score: ${(maxBiasScore * 100).toFixed(1)}%\n`;
    report += `Total Samples: ${totalSamples.toLocaleString()}\n\n`;
    
    report += `BIAS METRICS BY ATTRIBUTE\n`;
    report += `-------------------------\n`;
    biasMetrics.forEach(metric => {
        report += `\n${metric.protectedAttribute}:\n`;
        report += `  Demographic Parity Difference: ${(metric.demographicParityDifference * 100).toFixed(2)}%\n`;
        report += `  Equalized Odds Difference: ${(metric.equalizedOddsDifference * 100).toFixed(2)}%\n`;
        report += `  Accuracy Difference: ${(metric.accuracyDifference * 100).toFixed(2)}%\n`;
        report += `  Bias Severity: ${metric.biasSeverity}\n`;
    });
    
    return report;
}

function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    URL.revokeObjectURL(url);
}

// Utility functions
function showLoading(show) {
    const btn = document.getElementById('runAuditBtn');
    const btnText = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.loading-spinner');
    
    if (show) {
        btn.disabled = true;
        btnText.textContent = 'Running Audit...';
        spinner.style.display = 'inline-block';
    } else {
        btn.disabled = false;
        btnText.textContent = 'Run Bias Audit';
        spinner.style.display = 'none';
    }
}

function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toastContainer');
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    toastContainer.appendChild(toast);
    
    // Trigger animation
    setTimeout(() => toast.classList.add('show'), 100);
    
    // Remove toast after 3 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toastContainer.removeChild(toast), 300);
    }, 3000);
}

// Load existing results if available
function loadExistingResults() {
    // Check if we have existing CSV files to load
    fetch('bias_audit_results.csv')
        .then(response => {
            if (response.ok) {
                return response.text();
            }
            throw new Error('No existing results found');
        })
        .then(csvText => {
            Papa.parse(csvText, {
                header: true,
                skipEmptyLines: true,
                transform: function(value, field) {
                    // Convert numeric fields
                    if (['sample_size', 'accuracy', 'precision', 'recall', 'f1_score', 
                         'positive_prediction_rate', 'true_positive_rate', 'false_positive_rate'].includes(field)) {
                        return parseFloat(value) || 0;
                    }
                    return value;
                },
                complete: function(results) {
                    if (results.errors && results.errors.length > 0) {
                        console.log('CSV parsing errors:', results.errors);
                        return;
                    }
                    if (results.data && results.data.length > 0) {
                        // Filter out empty rows
                        auditResults = results.data.filter(row => 
                            row.protected_attribute && row.group
                        );
                        if (auditResults.length > 0) {
                            loadBiasSummary();
                        }
                    }
                }
            });
        })
        .catch(error => {
            console.log('No existing results to load');
        });
}

function loadBiasSummary() {
    fetch('bias_audit_results_summary.csv')
        .then(response => response.text())
        .then(csvText => {
            Papa.parse(csvText, {
                header: true,
                skipEmptyLines: true,
                transform: function(value, field) {
                    // Convert numeric fields
                    if (['demographic_parity_difference', 'equalized_odds_difference', 'accuracy_difference'].includes(field)) {
                        return parseFloat(value) || 0;
                    }
                    return value;
                },
                complete: function(results) {
                    if (results.errors && results.errors.length > 0) {
                        console.log('CSV parsing errors:', results.errors);
                        return;
                    }
                    if (results.data && results.data.length > 0) {
                        // Filter out empty rows
                        biasMetrics = results.data.filter(row => 
                            row.protected_attribute && row.bias_severity
                        );
                        if (biasMetrics.length > 0) {
                            displayResults();
                            showToast('Loaded existing bias audit results.', 'info');
                        }
                    }
                }
            });
        })
        .catch(error => {
            console.log('No existing bias summary to load');
        });
}

// Load sample data for testing
function loadSampleData() {
    showToast('Loading sample data...', 'info');
    
    // Clear previous results
    document.getElementById('resultsSection').style.display = 'none';
    
    fetch('sample_bias_data.csv')
        .then(response => {
            if (!response.ok) {
                throw new Error('Sample data not found');
            }
            return response.text();
        })
        .then(csvText => {
            Papa.parse(csvText, {
                header: true,
                skipEmptyLines: true,
                transformHeader: function(header) {
                    return header.trim();
                },
                complete: function(results) {
                    if (results.errors && results.errors.length > 0) {
                        console.warn('Sample data parsing warnings:', results.errors);
                    }
                    
                    if (!results.data || results.data.length === 0) {
                        showToast('Sample data appears to be empty.', 'error');
                        return;
                    }
                    
                    uploadedData = results.data.filter(row => 
                        Object.values(row).some(val => val !== null && val !== undefined && val !== '')
                    );
                    
                    if (uploadedData.length === 0) {
                        showToast('No valid data found in sample file.', 'error');
                        return;
                    }
                    
                    setupColumnConfiguration();
                    
                    // Auto-configure for sample data
                    setTimeout(() => {
                        document.getElementById('targetColumn').value = 'true_label';
                        document.getElementById('predictionColumn').value = 'model_prediction';
                        
                        // Check protected attributes
                        ['gender', 'race', 'education'].forEach(attr => {
                            const checkbox = document.getElementById(`attr_${attr}`);
                            if (checkbox) {
                                checkbox.checked = true;
                            }
                        });
                        
                        showToast(`Sample data loaded! ${uploadedData.length} records with suggested configuration.`, 'success');
                    }, 100);
                },
                error: function(error) {
                    showToast('Error loading sample data.', 'error');
                    console.error('Sample data error:', error);
                }
            });
        })
        .catch(error => {
            showToast('Sample data file not found. Please generate it first.', 'warning');
            console.error('Sample data fetch error:', error);
        });
} 