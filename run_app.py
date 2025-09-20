#!/usr/bin/env python3
"""
Startup script for Resume Relevance Check System
Handles initial setup and launches the Streamlit application
"""

import os
import sys
import subprocess
import importlib.util

def check_and_install_requirements():
    """Check if required packages are installed, install if missing"""
    
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'scikit-learn',
        'nltk',
        'PyPDF2',
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print("Installing missing packages...")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("All packages installed successfully!")
    else:
        print("All required packages are already installed.")

def setup_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        
        # Check if data is already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            print("NLTK data already available.")
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            print("NLTK data downloaded successfully!")
            
    except ImportError:
        print("NLTK not installed. Please install requirements first.")

def create_sample_files():
    """Create sample data files for testing"""
    
    # Sample job descriptions
    sample_jd_1 = """Data Science Position at Axion Ray

We are looking for a Data Analyst/Scientist to work with manufacturing data and AI solutions.

Required Skills:
- Python (Pandas, NumPy)
- Data analysis and visualization
- SQL for data processing
- Experience with manufacturing data
- Machine learning knowledge

Qualifications:
- Bachelor's degree in Engineering or related field
- 1+ years of experience in data analysis
- Understanding of manufacturing processes

Location: Remote/Hybrid"""

    sample_jd_2 = """Data Engineer Internship

6-month internship program with potential for permanent employment.

Required Skills:
- Python and Spark
- SQL for complex queries
- Data pipeline development
- DevOps knowledge
- Experience with streaming data

Qualifications:
- B.Tech, BE degree
- 2022 or earlier graduates preferred
- Strong programming skills

Location: Pune (Onsite)"""

    # Create samples directory if it doesn't exist
    if not os.path.exists('sample_data'):
        os.makedirs('sample_data')
    
    # Write sample JDs
    with open('sample_data/sample_jd_1.txt', 'w') as f:
        f.write(sample_jd_1)
    
    with open('sample_data/sample_jd_2.txt', 'w') as f:
        f.write(sample_jd_2)
    
    print("Sample data files created in 'sample_data' directory.")

def main():
    """Main setup and launch function"""
    
    print("🎯 Resume Relevance Check System - Setup")
    print("=" * 50)
    
    # Step 1: Check and install requirements
    print("\n1. Checking requirements...")
    check_and_install_requirements()
    
    # Step 2: Setup NLTK data
    print("\n2. Setting up NLTK data...")
    setup_nltk_data()
    
    # Step 3: Create sample files
    print("\n3. Creating sample data...")
    create_sample_files()
    
    # Step 4: Launch Streamlit app
    print("\n4. Launching the application...")
    print("🚀 Starting Streamlit server...")
    print("📱 The app will open in your default browser")
    print("🔗 Manual URL: http://localhost:8501")
    print("\n" + "=" * 50)
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "complete_web_app.py",
            "--server.address", "0.0.0.0",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        print("Try running manually: streamlit run complete_web_app.py")

if __name__ == "__main__":
    main()