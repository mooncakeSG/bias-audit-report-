#!/usr/bin/env python3
"""
Comprehensive Bias Audit Setup Script
====================================

This script sets up the environment and runs the comprehensive bias audit analysis.
It handles installation of required packages and execution of the Jupyter notebook.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        print("Continuing with available packages...")
        return False

def setup_jupyter():
    """Set up Jupyter environment"""
    print("📓 Setting up Jupyter environment...")
    try:
        subprocess.check_call([sys.executable, "-m", "jupyter", "kernelspec", "install-self", "--user"])
        print("✅ Jupyter kernel installed!")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Jupyter kernel setup failed, but continuing...")
        return False

def run_notebook():
    """Execute the comprehensive bias audit notebook"""
    print("🚀 Running comprehensive bias audit analysis...")
    notebook_file = "comprehensive_bias_audit.ipynb"
    
    if not os.path.exists(notebook_file):
        print(f"❌ Notebook file {notebook_file} not found!")
        return False
    
    try:
        # Try to run with nbconvert for automated execution
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            notebook_file
        ]
        subprocess.check_call(cmd)
        print("✅ Bias audit analysis completed successfully!")
        
        # Also create HTML report
        html_cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "html",
            notebook_file
        ]
        subprocess.check_call(html_cmd)
        print("📄 HTML report generated: comprehensive_bias_audit.html")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running notebook: {e}")
        print("💡 Try running manually: jupyter notebook comprehensive_bias_audit.ipynb")
        return False

def main():
    """Main setup and execution function"""
    print("🔍 COMPREHENSIVE BIAS AUDIT SETUP")
    print("=" * 50)
    
    # Check if sample data exists
    if not os.path.exists("sample_bias_data.csv"):
        print("❌ Sample data file not found!")
        print("Please ensure 'sample_bias_data.csv' exists in the current directory.")
        return False
    
    # Step 1: Install requirements
    install_success = install_requirements()
    
    # Step 2: Setup Jupyter
    jupyter_success = setup_jupyter()
    
    # Step 3: Run the analysis
    print("\n🎯 STARTING BIAS AUDIT ANALYSIS")
    print("=" * 50)
    
    analysis_success = run_notebook()
    
    # Summary
    print("\n📋 SETUP SUMMARY")
    print("=" * 30)
    print(f"Requirements Installation: {'✅' if install_success else '⚠️'}")
    print(f"Jupyter Setup: {'✅' if jupyter_success else '⚠️'}")
    print(f"Bias Audit Analysis: {'✅' if analysis_success else '❌'}")
    
    if analysis_success:
        print("\n🎉 BIAS AUDIT COMPLETED SUCCESSFULLY!")
        print("📊 Check the following outputs:")
        print("  • comprehensive_bias_audit.ipynb (executable notebook)")
        print("  • comprehensive_bias_audit.html (HTML report)")
        print("  • comprehensive_bias_dashboard.png (visualization)")
        print("  • mitigation_comparison.png (mitigation results)")
    else:
        print("\n⚠️  MANUAL EXECUTION REQUIRED")
        print("Run: jupyter notebook comprehensive_bias_audit.ipynb")
    
    return analysis_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 