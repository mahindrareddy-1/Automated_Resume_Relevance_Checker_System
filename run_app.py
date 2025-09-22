import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import json
import sqlite3
import re
import os
from typing import Dict, List, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PDF processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("PyPDF2 not available. Only text file uploads supported.")

# NLTK with fallback
try:
    import nltk
    from nltk.corpus import stopwords
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        NLTK_AVAILABLE = True
        stop_words = set(stopwords.words('english'))
    except:
        NLTK_AVAILABLE = False
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
except ImportError:
    NLTK_AVAILABLE = False
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

# Configure page
st.set_page_config(
    page_title="Resume Relevance Check System",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 10px;
    }
    .verdict-high { color: #28a745; font-weight: bold; }
    .verdict-medium { color: #ffc107; font-weight: bold; }
    .verdict-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class ResumeAnalyzer:
    def __init__(self):
        # Skill keywords database
        self.skills = [
            'python', 'java', 'javascript', 'sql', 'mysql', 'postgresql',
            'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
            'power bi', 'tableau', 'excel', 'statistics', 'machine learning',
            'data analysis', 'data visualization', 'react', 'angular', 'django',
            'flask', 'nodejs', 'mongodb', 'aws', 'azure', 'docker', 'git'
        ]
        
        self.education_keywords = [
            'bachelor', 'b.tech', 'btech', 'b.e', 'be', 'b.sc', 'bsc',
            'master', 'm.tech', 'mtech', 'm.sc', 'msc', 'mba'
        ]
        
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    
    def extract_text_from_pdf(self, file):
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            return ""
        
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    def clean_text(self, text):
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        return text.strip().lower()
    
    def extract_skills(self, text):
        """Extract skills from text"""
        text = self.clean_text(text)
        found_skills = []
        
        for skill in self.skills:
            if skill in text:
                found_skills.append(skill)
        
        return found_skills
    
    def extract_education(self, text):
        """Extract education information"""
        text = self.clean_text(text)
        found_education = []
        
        for edu in self.education_keywords:
            if edu in text:
                found_education.append(edu)
        
        return found_education
    
    def calculate_hard_score(self, resume_skills, resume_education, jd_skills):
        """Calculate keyword-based score"""
        if not jd_skills:
            return 50  # Default score if no skills specified
        
        skill_matches = len(set(resume_skills) & set(jd_skills))
        skill_score = (skill_matches / len(jd_skills)) * 70
        
        education_score = 20 if resume_education else 0
        
        return min(skill_score + education_score, 100)
    
    def calculate_semantic_score(self, resume_text, jd_text):
        """Calculate semantic similarity"""
        try:
            corpus = [resume_text, jd_text]
            tfidf_matrix = self.tfidf.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity * 100
        except:
            return 50
    
    def analyze_resume(self, resume_text, jd_text, filename=""):
        """Main analysis function"""
        resume_skills = self.extract_skills(resume_text)
        resume_education = self.extract_education(resume_text)
        jd_skills = self.extract_skills(jd_text)
        
        hard_score = self.calculate_hard_score(resume_skills, resume_education, jd_skills)
        semantic_score = self.calculate_semantic_score(resume_text, jd_text)
        
        final_score = (hard_score * 0.7) + (semantic_score * 0.3)
        
        if final_score >= 75:
            verdict = "High"
        elif final_score >= 50:
            verdict = "Medium"
        else:
            verdict = "Low"
        
        missing_skills = [skill for skill in jd_skills if skill not in resume_skills]
        
        return {
            'filename': filename,
            'relevance_score': round(final_score, 2),
            'hard_score': round(hard_score, 2),
            'semantic_score': round(semantic_score, 2),
            'verdict': verdict,
            'found_skills': resume_skills,
            'missing_skills': missing_skills,
            'timestamp': datetime.now().isoformat()
        }

@st.cache_resource
def get_analyzer():
    return ResumeAnalyzer()

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 Resume Relevance Check System</h1>
        <p>AI-powered resume evaluation | Innomatics Research Labs</p>
    </div>
    """, unsafe_allow_html=True)
    
    analyzer = get_analyzer()
    
    # Navigation
    tab1, tab2, tab3 = st.tabs(["Single Analysis", "Bulk Processing", "Dashboard"])
    
    with tab1:
        single_analysis_page(analyzer)
    
    with tab2:
        bulk_analysis_page(analyzer)
    
    with tab3:
        dashboard_page()

def single_analysis_page(analyzer):
    st.header("📋 Single Resume Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Resume Upload")
        uploaded_file = st.file_uploader(
            "Choose resume file",
            type=['pdf', 'txt'] if PDF_AVAILABLE else ['txt'],
            help="Upload PDF or TXT file"
        )
        
        if uploaded_file:
            st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    with col2:
        st.subheader("💼 Job Description")
        
        # Sample JDs
        sample_jds = {
            "Data Science Role": """Data Scientist Position
            
Required Skills:
- Python programming
- SQL database knowledge
- Data analysis and visualization
- Machine learning fundamentals
- Statistics knowledge
- Power BI or Tableau experience

Qualifications:
- Bachelor's degree in relevant field
- 2+ years of data analysis experience""",
            
            "Software Engineer Role": """Software Engineering Position
            
Required Skills:
- Python, Java, or JavaScript
- Web development frameworks
- Database knowledge (MySQL, PostgreSQL)
- Git version control
- API development

Qualifications:
- Bachelor's degree in Computer Science
- 1+ years of development experience"""
        }
        
        selected_jd = st.selectbox("Select sample JD:", ["Custom"] + list(sample_jds.keys()))
        
        if selected_jd != "Custom":
            jd_text = st.text_area("Job Description:", value=sample_jds[selected_jd], height=200)
        else:
            jd_text = st.text_area("Job Description:", height=200)
    
    if st.button("🔍 Analyze Resume", type="primary"):
        if uploaded_file and jd_text:
            with st.spinner("Analyzing..."):
                # Extract text
                if uploaded_file.type == "application/pdf" and PDF_AVAILABLE:
                    resume_text = analyzer.extract_text_from_pdf(uploaded_file)
                else:
                    resume_text = uploaded_file.read().decode('utf-8', errors='ignore')
                
                if not resume_text.strip():
                    st.error("Could not extract text from file")
                    return
                
                # Analyze
                result = analyzer.analyze_resume(resume_text, jd_text, uploaded_file.name)
                
                # Display results
                display_results(result)
        else:
            st.warning("Please upload resume and enter job description")

def display_results(result):
    st.header("📊 Analysis Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Relevance Score", f"{result['relevance_score']}/100")
    
    with col2:
        verdict_class = f"verdict-{result['verdict'].lower()}"
        st.markdown(f"**Verdict:** <span class='{verdict_class}'>{result['verdict']}</span>", 
                   unsafe_allow_html=True)
    
    with col3:
        st.metric("🎯 Hard Match", f"{result['hard_score']}/100")
    
    with col4:
        st.metric("🧠 Semantic Match", f"{result['semantic_score']}/100")
    
    # Skills analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Found Skills")
        if result['found_skills']:
            for skill in result['found_skills'][:10]:
                st.success(f"• {skill}")
        else:
            st.info("No matching skills found")
    
    with col2:
        st.subheader("❌ Missing Skills")
        if result['missing_skills']:
            for skill in result['missing_skills'][:10]:
                st.error(f"• {skill}")
        else:
            st.success("All skills matched!")
    
    # Chart
    fig = go.Figure(data=[
        go.Bar(
            x=['Hard Match', 'Semantic Match', 'Final Score'],
            y=[result['hard_score'], result['semantic_score'], result['relevance_score']],
            marker_color=['#ff7f0e', '#2ca02c', '#1f77b4']
        )
    ])
    
    fig.update_layout(title="Score Breakdown", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

def bulk_analysis_page(analyzer):
    st.header("📦 Bulk Processing")
    
    st.subheader("💼 Job Description")
    jd_text = st.text_area("Enter job description for bulk analysis:", height=150)
    
    st.subheader("📄 Upload Multiple Resumes")
    uploaded_files = st.file_uploader(
        "Choose resume files",
        type=['pdf', 'txt'] if PDF_AVAILABLE else ['txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded")
    
    if st.button("🔍 Process All", type="primary"):
        if uploaded_files and jd_text:
            results = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                # Extract text
                if file.type == "application/pdf" and PDF_AVAILABLE:
                    text = analyzer.extract_text_from_pdf(file)
                else:
                    text = file.read().decode('utf-8', errors='ignore')
                
                if text.strip():
                    result = analyzer.analyze_resume(text, jd_text, file.name)
                    results.append(result)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Display bulk results
            display_bulk_results(results)
        else:
            st.warning("Please provide job description and upload files")

def display_bulk_results(results):
    if not results:
        return
    
    st.subheader("📊 Bulk Results")
    
    # Summary
    avg_score = np.mean([r['relevance_score'] for r in results])
    high_count = len([r for r in results if r['verdict'] == 'High'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Processed", len(results))
    with col2:
        st.metric("📈 Avg Score", f"{avg_score:.1f}")
    with col3:
        st.metric("🎯 High Matches", high_count)
    
    # Results table
    df_data = []
    for r in results:
        df_data.append({
            'Resume': r['filename'],
            'Score': r['relevance_score'],
            'Verdict': r['verdict'],
            'Hard Match': r['hard_score'],
            'Semantic Match': r['semantic_score']
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # Download option
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download CSV",
        data=csv,
        file_name=f"bulk_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# First, add session state management to store analysis results
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []

def dashboard_page():
    st.header("📊 Dashboard")
    
    # Get actual results from session state
    results = st.session_state.analysis_results
    
    if not results:
        st.info("📋 No data available. Analyze some resumes first to see dashboard insights!")
        
        # Optional: Show sample dashboard layout with placeholder
        st.subheader("Dashboard Preview")
        st.write("After analyzing resumes, you'll see:")
        st.write("• Verdict distribution pie chart")
        st.write("• Score distribution histogram") 
        st.write("• Summary statistics")
        st.write("• Recent analysis results table")
        return
    
    # Calculate statistics from real data
    total_resumes = len(results)
    avg_score = np.mean([r['relevance_score'] for r in results])
    high_count = len([r for r in results if r['verdict'] == 'High'])
    medium_count = len([r for r in results if r['verdict'] == 'Medium'])
    low_count = len([r for r in results if r['verdict'] == 'Low'])
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Resumes", total_resumes)
    
    with col2:
        st.metric("📈 Average Score", f"{avg_score:.1f}/100")
    
    with col3:
        st.metric("✅ High Matches", high_count)
    
    with col4:
        st.metric("⚠️ Low Matches", low_count)
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        # Verdict distribution pie chart
        verdict_data = {
            'High': high_count,
            'Medium': medium_count, 
            'Low': low_count
        }
        
        # Only show non-zero categories
        verdict_filtered = {k: v for k, v in verdict_data.items() if v > 0}
        
        if verdict_filtered:
            fig_pie = px.pie(
                values=list(verdict_filtered.values()),
                names=list(verdict_filtered.keys()),
                title="Verdict Distribution",
                color_discrete_map={
                    'High': '#28a745',
                    'Medium': '#ffc107', 
                    'Low': '#dc3545'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Score distribution histogram
        scores = [r['relevance_score'] for r in results]
        fig_hist = px.histogram(
            x=scores,
            title="Score Distribution",
            nbins=min(10, len(scores)),  # Adjust bins based on data size
            labels={'x': 'Relevance Score', 'y': 'Count'}
        )
        fig_hist.update_traces(marker_color='#667eea')
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Recent results table
    st.subheader("📋 Recent Analysis Results")
    
    # Convert results to DataFrame for display
    df_data = []
    for r in results[-10:]:  # Show last 10 results
        df_data.append({
            'Resume': r['filename'],
            'Score': f"{r['relevance_score']}/100",
            'Verdict': r['verdict'],
            'Hard Match': f"{r['hard_score']:.1f}",
            'Semantic Match': f"{r['semantic_score']:.1f}",
            'Timestamp': r['timestamp'][:16] if 'timestamp' in r else 'N/A'
        })
    
    if df_data:
        df = pd.DataFrame(df_data)
        
        # Style the dataframe
        def highlight_verdict(val):
            if val == 'High':
                return 'background-color: #d4edda'
            elif val == 'Medium':
                return 'background-color: #fff3cd'  
            elif val == 'Low':
                return 'background-color: #f8d7da'
            return ''
        
        styled_df = df.style.applymap(highlight_verdict, subset=['Verdict'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Export functionality
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Additional insights section
    if len(results) > 1:
        st.subheader("📊 Additional Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top skills analysis
            all_skills = []
            for r in results:
                if 'found_skills' in r:
                    all_skills.extend(r['found_skills'])
            
            if all_skills:
                skill_counts = pd.Series(all_skills).value_counts().head(10)
                fig_skills = px.bar(
                    x=skill_counts.values,
                    y=skill_counts.index,
                    orientation='h',
                    title="Top Skills Found",
                    labels={'x': 'Frequency', 'y': 'Skills'}
                )
                st.plotly_chart(fig_skills, use_container_width=True)
        
        with col2:
            # Score trends (if timestamps available)
            if all('timestamp' in r for r in results):
                df_trends = pd.DataFrame(results)
                df_trends['date'] = pd.to_datetime(df_trends['timestamp']).dt.date
                
                if len(df_trends['date'].unique()) > 1:
                    daily_avg = df_trends.groupby('date')['relevance_score'].mean().reset_index()
                    
                    fig_trend = px.line(
                        daily_avg,
                        x='date',
                        y='relevance_score', 
                        title='Score Trends Over Time',
                        labels={'relevance_score': 'Avg Score', 'date': 'Date'}
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)

# Update the single analysis function to store results
def single_analysis_page(analyzer):
    st.header("📋 Single Resume Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Resume Upload")
        uploaded_file = st.file_uploader(
            "Choose resume file",
            type=['pdf', 'txt'] if PDF_AVAILABLE else ['txt'],
            help="Upload PDF or TXT file"
        )
        
        if uploaded_file:
            st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    with col2:
        st.subheader("💼 Job Description")
        
        # Sample JDs (same as before)
        sample_jds = {
            "Data Science Role": """Data Scientist Position
            
Required Skills:
- Python programming
- SQL database knowledge  
- Data analysis and visualization
- Machine learning fundamentals
- Statistics knowledge
- Power BI or Tableau experience

Qualifications:
- Bachelor's degree in relevant field
- 2+ years of data analysis experience""",
            
            "Software Engineer Role": """Software Engineering Position
            
Required Skills:
- Python, Java, or JavaScript
- Web development frameworks
- Database knowledge (MySQL, PostgreSQL)
- Git version control
- API development

Qualifications:
- Bachelor's degree in Computer Science
- 1+ years of development experience"""
        }
        
        selected_jd = st.selectbox("Select sample JD:", ["Custom"] + list(sample_jds.keys()))
        
        if selected_jd != "Custom":
            jd_text = st.text_area("Job Description:", value=sample_jds[selected_jd], height=200)
        else:
            jd_text = st.text_area("Job Description:", height=200)
    
    if st.button("🔍 Analyze Resume", type="primary"):
        if uploaded_file and jd_text:
            with st.spinner("Analyzing..."):
                # Extract text
                if uploaded_file.type == "application/pdf" and PDF_AVAILABLE:
                    resume_text = analyzer.extract_text_from_pdf(uploaded_file)
                else:
                    resume_text = uploaded_file.read().decode('utf-8', errors='ignore')
                
                if not resume_text.strip():
                    st.error("Could not extract text from file")
                    return
                
                # Analyze
                result = analyzer.analyze_resume(resume_text, jd_text, uploaded_file.name)
                
                # Store result in session state
                st.session_state.analysis_results.append(result)
                
                # Display results
                display_results(result)
                
                # Show success message
                st.success("✅ Analysis completed and saved to dashboard!")
        else:
            st.warning("Please upload resume and enter job description")

# Update bulk analysis to store results too
def bulk_analysis_page(analyzer):
    st.header("📦 Bulk Processing")
    
    st.subheader("💼 Job Description")
    jd_text = st.text_area("Enter job description for bulk analysis:", height=150)
    
    st.subheader("📄 Upload Multiple Resumes")
    uploaded_files = st.file_uploader(
        "Choose resume files",
        type=['pdf', 'txt'] if PDF_AVAILABLE else ['txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded")
    
    if st.button("🔍 Process All", type="primary"):
        if uploaded_files and jd_text:
            results = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                # Extract text
                if file.type == "application/pdf" and PDF_AVAILABLE:
                    text = analyzer.extract_text_from_pdf(file)
                else:
                    text = file.read().decode('utf-8', errors='ignore')
                
                if text.strip():
                    result = analyzer.analyze_resume(text, jd_text, file.name)
                    results.append(result)
                    # Add to session state
                    st.session_state.analysis_results.append(result)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Display bulk results
            display_bulk_results(results)
            
            st.success(f"✅ Processed {len(results)} resumes and saved to dashboard!")
        else:
            st.warning("Please provide job description and upload files")

if __name__ == "__main__":
    main()

