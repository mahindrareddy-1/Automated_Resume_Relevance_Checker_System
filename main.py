import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import json
import sqlite3
import re
import tempfile
import os
from typing import Dict, List, Tuple, Any
import logging

# Text processing libraries
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

# Configure Streamlit page
st.set_page_config(
    page_title="Resume Relevance Check System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .verdict-high { 
        color: #28a745; 
        font-weight: bold; 
        background-color: #d4edda; 
        padding: 4px 8px; 
        border-radius: 4px; 
    }
    .verdict-medium { 
        color: #856404; 
        font-weight: bold; 
        background-color: #fff3cd; 
        padding: 4px 8px; 
        border-radius: 4px; 
    }
    .verdict-low { 
        color: #721c24; 
        font-weight: bold; 
        background-color: #f8d7da; 
        padding: 4px 8px; 
        border-radius: 4px; 
    }
    .skill-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 4px;
        font-size: 0.8em;
    }
    .missing-skill-tag {
        display: inline-block;
        background-color: #ffebee;
        color: #d32f2f;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 4px;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

class ResumeParser:
    """Handles resume text extraction and parsing"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        self.stemmer = PorterStemmer()
        
        # Extended skill keywords
        self.skill_keywords = [
            'python', 'java', 'javascript', 'sql', 'mysql', 'postgresql',
            'react', 'angular', 'vue', 'node.js', 'express', 'django',
            'flask', 'spring', 'hibernate', 'mongodb', 'redis',
            'aws', 'azure', 'docker', 'kubernetes', 'git', 'jenkins',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch',
            'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
            'power bi', 'tableau', 'excel', 'statistics', 'data analysis',
            'eda', 'data visualization', 'web scraping', 'beautiful soup',
            'selenium', 'api', 'rest', 'graphql', 'html', 'css',
            'bootstrap', 'tailwind', 'figma', 'photoshop', 'r',
            'spark', 'kafka', 'hadoop', 'databricks', 'snowflake'
        ]
        
    def extract_text_from_pdf(self, file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error extracting PDF text: {e}")
            return ""
    
    def extract_text_from_docx(self, file) -> str:
        """Extract text from DOCX file (simplified version)"""
        try:
            # For simplicity, we'll ask user to convert to PDF or text
            st.warning("DOCX support is limited. Please convert to PDF or upload as text file.")
            return ""
        except Exception as e:
            st.error(f"Error extracting DOCX text: {e}")
            return ""
    
    def clean_and_normalize_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        return text.strip()
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.skill_keywords:
            if skill in text_lower:
                found_skills.append(skill)
        
        return list(set(found_skills))
    
    def extract_education(self, text: str) -> List[str]:
        """Extract education information"""
        education_patterns = [
            r'b\.?tech|bachelor of technology|btech',
            r'b\.?e\.?|bachelor of engineering',
            r'm\.?tech|master of technology|mtech',
            r'b\.?sc|bachelor of science|bsc',
            r'm\.?sc|master of science|msc',
            r'mba|master of business administration',
            r'b\.?ca|bachelor of computer application|bca',
            r'm\.?ca|master of computer application|mca',
        ]
        
        text_lower = text.lower()
        education = []
        
        for pattern in education_patterns:
            matches = re.findall(pattern, text_lower)
            education.extend(matches)
        
        return list(set(education))

class JobDescriptionParser:
    """Handles job description parsing and analysis"""
    
    def parse_jd(self, jd_text: str) -> Dict[str, Any]:
        """Parse job description and extract key information"""
        return {
            'role_title': self.extract_role_title(jd_text),
            'must_have_skills': self.extract_must_have_skills(jd_text),
            'good_to_have_skills': self.extract_good_to_have_skills(jd_text),
            'qualifications': self.extract_qualifications(jd_text),
            'experience': self.extract_experience(jd_text),
        }
    
    def extract_role_title(self, text: str) -> str:
        """Extract role title from JD"""
        job_titles = [
            'data scientist', 'data analyst', 'software engineer', 'full stack developer',
            'backend developer', 'frontend developer', 'machine learning engineer',
            'data engineer', 'business analyst', 'product manager', 'devops engineer'
        ]
        
        text_lower = text.lower()
        for title in job_titles:
            if title in text_lower:
                return title.title()
        
        return "Software Engineer"  # Default
    
    def extract_must_have_skills(self, text: str) -> List[str]:
        """Extract must-have skills from JD"""
        common_skills = [
            'python', 'java', 'javascript', 'sql', 'mysql', 'postgresql',
            'react', 'angular', 'vue', 'node.js', 'express', 'django',
            'flask', 'pandas', 'numpy', 'machine learning', 'data analysis',
            'statistics', 'excel', 'power bi', 'tableau', 'spark', 'kafka'
        ]
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in common_skills:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def extract_good_to_have_skills(self, text: str) -> List[str]:
        """Extract good-to-have skills"""
        return []  # Simplified for demo
    
    def extract_qualifications(self, text: str) -> List[str]:
        """Extract qualification requirements"""
        qualifications = []
        text_lower = text.lower()
        
        if 'bachelor' in text_lower or 'b.tech' in text_lower:
            qualifications.append('Bachelor\'s degree')
        if 'master' in text_lower or 'm.tech' in text_lower:
            qualifications.append('Master\'s degree')
            
        return qualifications
    
    def extract_experience(self, text: str) -> str:
        """Extract experience requirements"""
        exp_patterns = [
            r'(\d+)\s*(?:to|\-)\s*(\d+)\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return "1+ years"

class RelevanceAnalyzer:
    """Main class for analyzing resume relevance"""
    
    def __init__(self):
        self.resume_parser = ResumeParser()
        self.jd_parser = JobDescriptionParser()
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
    
    def analyze_resume(self, resume_text: str, jd_text: str, resume_filename: str = "") -> Dict[str, Any]:
        """Main function to analyze resume against job description"""
        
        resume_skills = self.resume_parser.extract_skills(resume_text)
        resume_education = self.resume_parser.extract_education(resume_text)
        jd_data = self.jd_parser.parse_jd(jd_text)
        
        hard_match_score = self.calculate_hard_match_score(resume_skills, resume_education, jd_data)
        semantic_score = self.calculate_semantic_score(resume_text, jd_text)
        final_score = self.calculate_final_score(hard_match_score, semantic_score)
        verdict = self.generate_verdict(final_score)
        missing_skills = self.identify_missing_skills(resume_skills, jd_data)
        suggestions = self.generate_suggestions(missing_skills, jd_data)
        
        return {
            'resume_filename': resume_filename,
            'relevance_score': round(final_score, 2),
            'verdict': verdict,
            'hard_match_score': round(hard_match_score, 2),
            'semantic_score': round(semantic_score, 2),
            'job_role': jd_data['role_title'],
            'found_skills': resume_skills,
            'missing_skills': missing_skills,
            'suggestions': suggestions,
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_hard_match_score(self, resume_skills: List[str], resume_education: List[str], jd_data: Dict) -> float:
        """Calculate hard match score"""
        total_score = 0
        
        # Skills matching (70% weight)
        must_have_skills = jd_data['must_have_skills']
        if must_have_skills:
            matches = len(set(resume_skills) & set(must_have_skills))
            skill_score = (matches / len(must_have_skills)) * 70
            total_score += skill_score
        
        # Education matching (30% weight)
        if jd_data['qualifications'] and resume_education:
            total_score += 30
        elif resume_education:
            total_score += 15
        
        return min(total_score, 100)
    
    def calculate_semantic_score(self, resume_text: str, jd_text: str) -> float:
        """Calculate semantic similarity score using TF-IDF"""
        try:
            corpus = [resume_text, jd_text]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity * 100
        except:
            return 50.0
    
    def calculate_final_score(self, hard_score: float, semantic_score: float) -> float:
        """Calculate weighted final score"""
        return (hard_score * 0.7) + (semantic_score * 0.3)
    
    def generate_verdict(self, score: float) -> str:
        """Generate hiring verdict based on score"""
        if score >= 75:
            return "High"
        elif score >= 50:
            return "Medium"
        else:
            return "Low"
    
    def identify_missing_skills(self, resume_skills: List[str], jd_data: Dict) -> List[str]:
        """Identify missing skills"""
        missing = []
        for skill in jd_data['must_have_skills']:
            if skill not in resume_skills:
                missing.append(skill)
        return missing
    
    def generate_suggestions(self, missing_skills: List[str], jd_data: Dict) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        if missing_skills:
            suggestions.append(f"Consider gaining experience in: {', '.join(missing_skills[:5])}")
        suggestions.append("Add more relevant projects showcasing your technical skills")
        suggestions.append("Include certifications related to the job requirements")
        return suggestions

class DatabaseManager:
    """Handles database operations"""
    
    def __init__(self, db_path: str = "resume_analysis.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resume_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_filename TEXT NOT NULL,
                job_role TEXT NOT NULL,
                relevance_score REAL NOT NULL,
                verdict TEXT NOT NULL,
                hard_match_score REAL,
                semantic_score REAL,
                found_skills TEXT,
                missing_skills TEXT,
                suggestions TEXT,
                timestamp TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_analysis(self, result: Dict[str, Any]) -> int:
        """Save analysis result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO resume_analysis 
            (resume_filename, job_role, relevance_score, verdict, hard_match_score, 
             semantic_score, found_skills, missing_skills, suggestions, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['resume_filename'],
            result['job_role'],
            result['relevance_score'],
            result['verdict'],
            result['hard_match_score'],
            result['semantic_score'],
            json.dumps(result['found_skills']),
            json.dumps(result['missing_skills']),
            json.dumps(result['suggestions']),
            result['timestamp']
        ))
        
        result_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return result_id
    
    def get_all_results(self) -> List[Dict]:
        """Get all analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM resume_analysis ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'resume_filename': row[1],
                'job_role': row[2],
                'relevance_score': row[3],
                'verdict': row[4],
                'hard_match_score': row[5],
                'semantic_score': row[6],
                'found_skills': json.loads(row[7]) if row[7] else [],
                'missing_skills': json.loads(row[8]) if row[8] else [],
                'suggestions': json.loads(row[9]) if row[9] else [],
                'timestamp': row[10]
            })
        
        conn.close()
        return results

# Initialize components
@st.cache_resource
def get_analyzer():
    return RelevanceAnalyzer()

@st.cache_resource
def get_db_manager():
    return DatabaseManager()

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Automated Resume Relevance Check System</h1>
        <p>AI-powered resume evaluation against job requirements | Innomatics Research Labs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    analyzer = get_analyzer()
    db_manager = get_db_manager()
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🔍 Resume Analysis", 
        "📊 Dashboard", 
        "📦 Bulk Upload", 
        "📈 Results History"
    ])
    
    if page == "🔍 Resume Analysis":
        show_analysis_page(analyzer, db_manager)
    elif page == "📊 Dashboard":
        show_dashboard_page(db_manager)
    elif page == "📦 Bulk Upload":
        show_bulk_upload_page(analyzer, db_manager)
    elif page == "📈 Results History":
        show_results_history_page(db_manager)

def show_analysis_page(analyzer, db_manager):
    """Single resume analysis page"""
    st.header("📋 Single Resume Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Upload Resume")
        uploaded_resume = st.file_uploader(
            "Choose resume file",
            type=['pdf', 'txt'],
            help="Upload PDF or TXT files only"
        )
        
        if uploaded_resume:
            st.success(f"✅ Resume uploaded: {uploaded_resume.name}")
    
    with col2:
        st.subheader("💼 Job Description")
        
        # Sample JDs for quick testing
        sample_jds = {
            "Data Science Role": """We are looking for a Data Analyst with experience in Python, SQL, and data visualization.
            
Required Skills:
- Python (Pandas, NumPy)
- SQL (MySQL, PostgreSQL) 
- Data visualization tools (Power BI, Tableau)
- Statistics and data analysis
- Machine learning basics

Qualifications:
- Bachelor's degree in relevant field
- 2+ years of experience in data analysis""",
            
            "Software Engineering Role": """We are seeking a Software Engineer to join our development team.

Required Skills:
- Python, Java, or JavaScript
- Web development frameworks (React, Angular, Django)
- Database knowledge (MySQL, PostgreSQL)
- Git version control
- API development experience

Qualifications:
- Bachelor's degree in Computer Science or related field
- 1+ years of software development experience"""
        }
        
        selected_sample = st.selectbox("Select sample JD:", ["Custom"] + list(sample_jds.keys()))
        
        if selected_sample != "Custom":
            jd_text = st.text_area("Job description:", value=sample_jds[selected_sample], height=200)
        else:
            jd_text = st.text_area("Job description:", height=200, 
                                 placeholder="Enter the complete job description...")
    
    # Analysis button
    if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
        if uploaded_resume and jd_text:
            with st.spinner("Analyzing resume... Please wait."):
                try:
                    # Extract text from resume
                    if uploaded_resume.type == "application/pdf":
                        resume_text = analyzer.resume_parser.extract_text_from_pdf(uploaded_resume)
                    else:
                        resume_text = uploaded_resume.read().decode('utf-8', errors='ignore')
                    
                    if not resume_text.strip():
                        st.error("Could not extract text from the resume. Please try a different file.")
                        return
                    
                    # Analyze resume
                    result = analyzer.analyze_resume(resume_text, jd_text, uploaded_resume.name)
                    
                    # Save to database
                    db_manager.save_analysis(result)
                    
                    # Display results
                    display_analysis_result(result)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        else:
            st.warning("⚠️ Please upload both resume and job description.")

def display_analysis_result(result):
    """Display analysis results"""
    st.header("📊 Analysis Results")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Relevance Score", f"{result['relevance_score']}/100")
    
    with col2:
        verdict_class = f"verdict-{result['verdict'].lower()}"
        st.markdown(f"**Verdict:** <span class='{verdict_class}'>{result['verdict']}</span>", 
                   unsafe_allow_html=True)
    
    with col3:
        st.metric("🎯 Hard Match", f"{result['hard_match_score']:.1f}/100")
    
    with col4:
        st.metric("🧠 Semantic Match", f"{result['semantic_score']:.1f}/100")
    
    # Skills analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Found Skills")
        if result['found_skills']:
            skills_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in result['found_skills'][:15]])
            st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.info("No matching skills found")
    
    with col2:
        st.subheader("❌ Missing Skills")
        if result['missing_skills']:
            missing_html = "".join([f'<span class="missing-skill-tag">{skill}</span>' for skill in result['missing_skills'][:15]])
            st.markdown(missing_html, unsafe_allow_html=True)
        else:
            st.success("All required skills found!")
    
    # Suggestions
    st.subheader("💡 Improvement Suggestions")
    for i, suggestion in enumerate(result['suggestions'], 1):
        st.info(f"{i}. {suggestion}")
    
    # Score breakdown chart
    fig = go.Figure(data=[
        go.Bar(
            x=['Hard Match', 'Semantic Match', 'Final Score'], 
            y=[result['hard_match_score'], result['semantic_score'], result['relevance_score']],
            marker_color=['#ff7f0e', '#2ca02c', '#1f77b4'],
            text=[f"{result['hard_match_score']:.1f}", f"{result['semantic_score']:.1f}", f"{result['relevance_score']:.1f}"],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Score Breakdown Analysis",
        yaxis_title="Score (0-100)",
        showlegend=False,
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_dashboard_page(db_manager):
    """Dashboard overview page"""
    st.header("📊 Dashboard Overview")
    
    results = db_manager.get_all_results()
    
    if not results:
        st.info("No resumes analyzed yet. Please analyze some resumes first!")
        return
    
    # Key metrics
    high_matches = len([r for r in results if r['verdict'] == 'High'])
    medium_matches = len([r for r in results if r['verdict'] == 'Medium'])
    low_matches = len([r for r in results if r['verdict'] == 'Low'])
    avg_score = np.mean([r['relevance_score'] for r in results])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Resumes", len(results))
    
    with col2:
        st.metric("🎯 Average Score", f"{avg_score:.1f}/100")
    
    with col3:
        st.metric("✅ High Matches", high_matches)
    
    with col4:
        st.metric("⚠️ Low Matches", low_matches)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Verdict distribution pie chart
        fig_pie = px.pie(
            values=[high_matches, medium_matches, low_matches],
            names=['High', 'Medium', 'Low'],
            title="Resume Distribution by Verdict",
            color_discrete_map={'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Score distribution histogram
        scores = [r['relevance_score'] for r in results]
        fig_hist = px.histogram(
            x=scores,
            title="Score Distribution",
            nbins=10,
            labels={'x': 'Relevance Score', 'y': 'Count'}
        )
        fig_hist.update_traces(marker_color='#667eea')
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Recent results
    st.subheader("📋 Recent Analysis Results")
    recent_results = results[:10]
    
    df_data = []
    for r in recent_results:
        df_data.append({
            'Resume': r['resume_filename'],
            'Job Role': r['job_role'],
            'Score': f"{r['relevance_score']}/100",
            'Verdict': r['verdict'],
            'Date': r['timestamp'][:16]
        })
    
    if df_data:
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)

def show_bulk_upload_page(analyzer, db_manager):
    """Bulk upload page"""
    st.header("📦 Bulk Resume Upload")
    st.write("Upload multiple resumes to analyze against a single job description.")
    
    # Job description
    st.subheader("💼 Job Description")
    jd_text = st.text_area("Enter job description:", height=150)
    
    # File upload
    st.subheader("📄 Upload Multiple Resumes")
    uploaded_files = st.file_uploader(
        "Choose resume files",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded")
    
    if st.button("🔍 Analyze All Resumes", type="primary"):
        if uploaded_files and jd_text:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}... ({i+1}/{len(uploaded_files)})")
                
                try:
                    if file.type == "application/pdf":
                        resume_text = analyzer.resume_parser.extract_text_from_pdf(file)
                    else:
                        resume_text = file.read().decode('utf-8', errors='ignore')
                    
                    if resume_text.strip():
                        result = analyzer.analyze_resume(resume_text, jd_text, file.name)
                        db_manager.save_analysis(result)
                        results.append(result)
                    else:
                        st.warning(f"Could not extract text from {file.name}")
                        
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ Analysis complete!")
            display_bulk_results(results)
        else:
            st.warning("⚠️ Please provide job description and upload files.")

def display_bulk_results(results):
    """Display bulk analysis results"""
    if not results:
        return
    
    st.header("📊 Bulk Analysis Results")
    
    # Summary metrics
    avg_score = np.mean([r['relevance_score'] for r in results])
    high_count = len([r for r in results if r['verdict'] == 'High'])
    top_score = max([r['relevance_score'] for r in results])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Processed", len(results))
    with col2:
        st.metric("📈 Avg Score", f"{avg_score:.1f}")
    with col3:
        st.metric("🎯 High Matches", high_count)
    with col4:
        st.metric("🏆 Top Score", f"{top_score:.1f}")
    
    # Results table
    df_data = []
    for r in results:
        df_data.append({
            'Resume': r['resume_filename'],
            'Score': f"{r['relevance_score']}/100",
            'Verdict': r['verdict'],
            'Hard Match': f"{r['hard_match_score']:.1f}",
            'Semantic Match': f"{r['semantic_score']:.1f}"
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # Download option
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        "📥 Download Results as CSV",
        data=csv_buffer.getvalue(),
        file_name=f"bulk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def show_results_history_page(db_manager):
    """Results history page"""
    st.header("📈 Results History & Analytics")
    
    # Get all results
    all_results = db_manager.get_all_results()
    
    if not all_results:
        st.info("No analysis history available.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_score = st.slider("Minimum Score", 0, 100, 0)
    
    with col2:
        verdict_filter = st.selectbox("Verdict", ["All", "High", "Medium", "Low"])
    
    with col3:
        job_roles = list(set([r['job_role'] for r in all_results]))
        role_filter = st.selectbox("Job Role", ["All"] + job_roles)
    
    # Apply filters
    filtered_results = all_results
    
    if min_score > 0:
        filtered_results = [r for r in filtered_results if r['relevance_score'] >= min_score]
    
    if verdict_filter != "All":
        filtered_results = [r for r in filtered_results if r['verdict'] == verdict_filter]
    
    if role_filter != "All":
        filtered_results = [r for r in filtered_results if r['job_role'] == role_filter]
    
    st.subheader(f"Found {len(filtered_results)} results")
    
    if not filtered_results:
        st.warning("No results match the selected filters.")
        return
    
    # Analytics charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Score trends over time
        df_trends = pd.DataFrame(filtered_results)
        df_trends['date'] = pd.to_datetime(df_trends['timestamp']).dt.date
        daily_avg = df_trends.groupby('date')['relevance_score'].mean().reset_index()
        
        if len(daily_avg) > 1:
            fig_trend = px.line(
                daily_avg,
                x='date',
                y='relevance_score',
                title='Average Score Trend',
                labels={'relevance_score': 'Avg Score', 'date': 'Date'}
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Need more data points for trend analysis")
    
    with col2:
        # Job role distribution
        role_counts = pd.Series([r['job_role'] for r in filtered_results]).value_counts()
        
        if len(role_counts) > 0:
            fig_roles = px.bar(
                x=role_counts.values,
                y=role_counts.index,
                orientation='h',
                title='Job Role Distribution'
            )
            st.plotly_chart(fig_roles, use_container_width=True)
    
    # Results table
    st.subheader("Detailed Results")
    
    df_history = pd.DataFrame([{
        'Resume': r['resume_filename'],
        'Job Role': r['job_role'],
        'Score': r['relevance_score'],
        'Verdict': r['verdict'],
        'Date': r['timestamp'][:16]
    } for r in filtered_results])
    
    st.dataframe(df_history, use_container_width=True)

# Sample data for testing
def load_sample_data():
    """Load sample resumes and JDs for testing"""
    
    sample_resumes = {
        "Data Analyst Resume 1": """Pavan Kalyan
Data Analyst with experience in Python, SQL, and Power BI
Skills: Python, Pandas, NumPy, SQL, MySQL, Power BI, Tableau, Excel
Education: Bachelor of Science in Physics
Projects: Data analysis on used car listings, Pizza Hut sales analysis
Certifications: Advanced Data Science with Python""",
        
        "Software Developer Resume 1": """Jay Raj
Aspiring Software Developer with Python and web development skills
Skills: Python, JavaScript, HTML, CSS, React, Node.js, MongoDB
Education: Bachelor of Technology in Computer Science
Projects: E-commerce website, Weather app, Task manager""",
        
        "Business Analyst Resume 1": """Suresh Shiv
Business Analyst with SQL and Power BI expertise
Skills: SQL, Power BI, Excel, Python, Data Visualization, Business Intelligence
Education: Bachelor of Engineering in Electronics
Projects: Sales dashboard, Market analysis, Customer segmentation"""
    }
    
    return sample_resumes

def show_sample_data_section():
    """Show sample data for testing"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("🧪 Quick Test")
        
        if st.button("Load Sample Data"):
            sample_resumes = load_sample_data()
            
            st.success("Sample data loaded! You can copy and paste these for testing:")
            
            for name, content in sample_resumes.items():
                with st.expander(name):
                    st.text_area("Content:", content, height=100, key=f"sample_{name}")

# Main execution
if __name__ == "__main__":
    # Add sample data section to sidebar
    show_sample_data_section()
    
    # Run main app
    main()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎯 Resume Relevance Check System | Built for Innomatics Research Labs</p>
        <p>Powered by AI • Python • Streamlit</p>
    </div>
    """, unsafe_allow_html=True)