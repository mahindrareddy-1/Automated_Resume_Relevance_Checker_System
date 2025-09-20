import os
import json
import re
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging

# Text processing libraries
import PyPDF2
import docx2txt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Data processing
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy

# Fuzzy matching
from fuzzywuzzy import fuzz, process

# Web framework
import streamlit as st
from io import BytesIO
import tempfile

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeParser:
    """Handles resume text extraction and parsing"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
    def extract_text_from_pdf(self, file_path_or_bytes) -> str:
        """Extract text from PDF file"""
        try:
            if isinstance(file_path_or_bytes, str):
                with open(file_path_or_bytes, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
            else:
                pdf_reader = PyPDF2.PdfReader(file_path_or_bytes)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path_or_bytes) -> str:
        """Extract text from DOCX file"""
        try:
            if isinstance(file_path_or_bytes, str):
                return docx2txt.process(file_path_or_bytes)
            else:
                # For bytes, save temporarily
                with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp_file:
                    tmp_file.write(file_path_or_bytes.read())
                    tmp_file.flush()
                    text = docx2txt.process(tmp_file.name)
                os.unlink(tmp_file.name)
                return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ""
    
    def clean_and_normalize_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        text = text.strip()
        return text
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        # Common skill keywords (this should be expanded)
        skill_keywords = [
            'python', 'java', 'javascript', 'sql', 'mysql', 'postgresql',
            'react', 'angular', 'vue', 'node.js', 'express', 'django',
            'flask', 'spring', 'hibernate', 'mongodb', 'redis',
            'aws', 'azure', 'docker', 'kubernetes', 'git', 'jenkins',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch',
            'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
            'power bi', 'tableau', 'excel', 'statistics', 'data analysis',
            'eda', 'data visualization', 'web scraping', 'beautiful soup',
            'selenium', 'api', 'rest', 'graphql', 'html', 'css',
            'bootstrap', 'tailwind', 'figma', 'photoshop'
        ]
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in skill_keywords:
            if skill in text_lower:
                found_skills.append(skill)
        
        return list(set(found_skills))
    
    def extract_education(self, text: str) -> List[str]:
        """Extract education information"""
        education_patterns = [
            r'b\.?tech|bachelor of technology|btech',
            r'b\.?e\.?|bachelor of engineering',
            r'm\.?tech|master of technology|mtech',
            r'm\.?e\.?|master of engineering',
            r'b\.?sc|bachelor of science|bsc',
            r'm\.?sc|master of science|msc',
            r'mba|master of business administration',
            r'b\.?com|bachelor of commerce|bcom',
            r'm\.?com|master of commerce|mcom',
            r'b\.?ca|bachelor of computer application|bca',
            r'm\.?ca|master of computer application|mca',
            r'phd|ph\.d|doctorate'
        ]
        
        text_lower = text.lower()
        education = []
        
        for pattern in education_patterns:
            matches = re.findall(pattern, text_lower)
            education.extend(matches)
        
        return list(set(education))

class JobDescriptionParser:
    """Handles job description parsing and analysis"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def parse_jd(self, jd_text: str) -> Dict[str, Any]:
        """Parse job description and extract key information"""
        jd_data = {
            'role_title': self.extract_role_title(jd_text),
            'must_have_skills': self.extract_must_have_skills(jd_text),
            'good_to_have_skills': self.extract_good_to_have_skills(jd_text),
            'qualifications': self.extract_qualifications(jd_text),
            'experience': self.extract_experience(jd_text),
            'location': self.extract_location(jd_text)
        }
        return jd_data
    
    def extract_role_title(self, text: str) -> str:
        """Extract role title from JD"""
        # Look for common role patterns
        role_patterns = [
            r'(?i)position[:\s]+([^\n\.]+)',
            r'(?i)role[:\s]+([^\n\.]+)',
            r'(?i)job title[:\s]+([^\n\.]+)',
            r'(?i)we are looking for[:\s]+([^\n\.]+)',
            r'(?i)hiring[:\s]+([^\n\.]+)'
        ]
        
        for pattern in role_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # Default extraction from common job titles
        job_titles = [
            'data scientist', 'data analyst', 'software engineer', 'full stack developer',
            'backend developer', 'frontend developer', 'machine learning engineer',
            'data engineer', 'business analyst', 'product manager', 'devops engineer'
        ]
        
        text_lower = text.lower()
        for title in job_titles:
            if title in text_lower:
                return title.title()
        
        return "Not specified"
    
    def extract_must_have_skills(self, text: str) -> List[str]:
        """Extract must-have skills from JD"""
        must_have_patterns = [
            r'(?i)required skills?[:\s]*([^\n\.]+)',
            r'(?i)must have[:\s]*([^\n\.]+)',
            r'(?i)mandatory[:\s]*([^\n\.]+)',
            r'(?i)essential skills?[:\s]*([^\n\.]+)'
        ]
        
        skills = []
        for pattern in must_have_patterns:
            matches = re.findall(pattern, text)
            skills.extend(matches)
        
        # Extract from bullet points and common skill mentions
        text_lower = text.lower()
        common_skills = [
            'python', 'java', 'javascript', 'sql', 'mysql', 'postgresql',
            'react', 'angular', 'vue', 'node.js', 'express', 'django',
            'flask', 'pandas', 'numpy', 'machine learning', 'data analysis',
            'statistics', 'excel', 'power bi', 'tableau'
        ]
        
        for skill in common_skills:
            if skill in text_lower:
                skills.append(skill)
        
        return list(set([skill.strip() for skill in skills if skill.strip()]))
    
    def extract_good_to_have_skills(self, text: str) -> List[str]:
        """Extract good-to-have skills from JD"""
        good_to_have_patterns = [
            r'(?i)preferred[:\s]*([^\n\.]+)',
            r'(?i)good to have[:\s]*([^\n\.]+)',
            r'(?i)nice to have[:\s]*([^\n\.]+)',
            r'(?i)bonus[:\s]*([^\n\.]+)'
        ]
        
        skills = []
        for pattern in good_to_have_patterns:
            matches = re.findall(pattern, text)
            skills.extend(matches)
        
        return list(set([skill.strip() for skill in skills if skill.strip()]))
    
    def extract_qualifications(self, text: str) -> List[str]:
        """Extract qualification requirements"""
        qual_patterns = [
            r'(?i)bachelor|b\.?tech|b\.?e\.?|b\.?sc|graduation',
            r'(?i)master|m\.?tech|m\.?e\.?|m\.?sc|post.?graduation',
            r'(?i)mba|phd|doctorate'
        ]
        
        qualifications = []
        for pattern in qual_patterns:
            matches = re.findall(pattern, text)
            qualifications.extend(matches)
        
        return list(set(qualifications))
    
    def extract_experience(self, text: str) -> str:
        """Extract experience requirements"""
        exp_patterns = [
            r'(\d+)\s*(?:to|\-)\s*(\d+)\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'minimum\s*(\d+)\s*years?',
            r'at least\s*(\d+)\s*years?'
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return "Not specified"
    
    def extract_location(self, text: str) -> str:
        """Extract job location"""
        location_patterns = [
            r'(?i)location[:\s]*([^\n\.]+)',
            r'(?i)based in[:\s]*([^\n\.]+)',
            r'(?i)office[:\s]*([^\n\.]+)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # Common locations
        locations = ['hyderabad', 'bangalore', 'pune', 'delhi', 'mumbai', 'chennai', 'kolkata']
        text_lower = text.lower()
        for location in locations:
            if location in text_lower:
                return location.title()
        
        return "Not specified"

class RelevanceAnalyzer:
    """Main class for analyzing resume relevance against job descriptions"""
    
    def __init__(self):
        self.resume_parser = ResumeParser()
        self.jd_parser = JobDescriptionParser()
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            logger.warning("Could not load sentence transformer model. Semantic matching will be limited.")
            self.sentence_model = None
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
    
    def analyze_resume(self, resume_text: str, jd_text: str, resume_filename: str = "") -> Dict[str, Any]:
        """Main function to analyze resume against job description"""
        
        # Parse resume and JD
        resume_skills = self.resume_parser.extract_skills(resume_text)
        resume_education = self.resume_parser.extract_education(resume_text)
        jd_data = self.jd_parser.parse_jd(jd_text)
        
        # Perform hard match analysis
        hard_match_score = self.calculate_hard_match_score(
            resume_skills, resume_education, jd_data
        )
        
        # Perform semantic match analysis
        semantic_score = self.calculate_semantic_score(resume_text, jd_text)
        
        # Calculate final weighted score
        final_score = self.calculate_final_score(hard_match_score, semantic_score)
        
        # Generate verdict
        verdict = self.generate_verdict(final_score)
        
        # Identify missing skills
        missing_skills = self.identify_missing_skills(resume_skills, jd_data)
        
        # Generate improvement suggestions
        suggestions = self.generate_suggestions(missing_skills, jd_data)
        
        # Prepare result
        result = {
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
        
        return result
    
    def calculate_hard_match_score(self, resume_skills: List[str], resume_education: List[str], jd_data: Dict) -> float:
        """Calculate hard match score based on keywords and skills"""
        
        total_score = 0
        weights = {
            'must_have_skills': 0.4,
            'good_to_have_skills': 0.2,
            'qualifications': 0.3,
            'general_match': 0.1
        }
        
        # Must-have skills matching
        must_have_skills = [skill.lower() for skill in jd_data['must_have_skills']]
        must_have_matches = 0
        if must_have_skills:
            for skill in resume_skills:
                # Use fuzzy matching for better results
                best_match = process.extractOne(skill.lower(), must_have_skills)
                if best_match and best_match[1] > 80:  # 80% similarity threshold
                    must_have_matches += 1
            
            must_have_score = (must_have_matches / len(must_have_skills)) * 100
            total_score += must_have_score * weights['must_have_skills']
        
        # Good-to-have skills matching
        good_to_have_skills = [skill.lower() for skill in jd_data['good_to_have_skills']]
        good_to_have_matches = 0
        if good_to_have_skills:
            for skill in resume_skills:
                best_match = process.extractOne(skill.lower(), good_to_have_skills)
                if best_match and best_match[1] > 80:
                    good_to_have_matches += 1
            
            good_to_have_score = min((good_to_have_matches / len(good_to_have_skills)) * 100, 100)
            total_score += good_to_have_score * weights['good_to_have_skills']
        
        # Qualifications matching
        jd_qualifications = [qual.lower() for qual in jd_data['qualifications']]
        qual_matches = 0
        if jd_qualifications:
            for edu in resume_education:
                for qual in jd_qualifications:
                    if fuzz.partial_ratio(edu.lower(), qual) > 70:
                        qual_matches += 1
                        break
            
            qual_score = min((qual_matches / len(jd_qualifications)) * 100, 100)
            total_score += qual_score * weights['qualifications']
        
        # General skill overlap
        all_jd_skills = must_have_skills + good_to_have_skills
        if all_jd_skills:
            general_matches = len(set(resume_skills) & set(all_jd_skills))
            general_score = min((general_matches / len(set(all_jd_skills))) * 100, 100)
            total_score += general_score * weights['general_match']
        
        return min(total_score, 100)
    
    def calculate_semantic_score(self, resume_text: str, jd_text: str) -> float:
        """Calculate semantic similarity score"""
        
        if not self.sentence_model:
            # Fallback to TF-IDF based similarity
            return self.calculate_tfidf_similarity(resume_text, jd_text)
        
        try:
            # Generate embeddings
            resume_embedding = self.sentence_model.encode([resume_text])
            jd_embedding = self.sentence_model.encode([jd_text])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(resume_embedding, jd_embedding)[0][0]
            
            # Convert to percentage
            return similarity * 100
            
        except Exception as e:
            logger.error(f"Error in semantic scoring: {e}")
            return self.calculate_tfidf_similarity(resume_text, jd_text)
    
    def calculate_tfidf_similarity(self, resume_text: str, jd_text: str) -> float:
        """Fallback TF-IDF based similarity calculation"""
        try:
            # Fit TF-IDF on both texts
            corpus = [resume_text, jd_text]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity * 100
            
        except Exception as e:
            logger.error(f"Error in TF-IDF similarity: {e}")
            return 50.0  # Default score
    
    def calculate_final_score(self, hard_score: float, semantic_score: float) -> float:
        """Calculate weighted final score"""
        # Give more weight to hard matching for technical accuracy
        weights = {
            'hard_match': 0.7,
            'semantic_match': 0.3
        }
        
        final_score = (hard_score * weights['hard_match']) + (semantic_score * weights['semantic_match'])
        return min(final_score, 100)
    
    def generate_verdict(self, score: float) -> str:
        """Generate hiring verdict based on score"""
        if score >= 75:
            return "High"
        elif score >= 50:
            return "Medium"
        else:
            return "Low"
    
    def identify_missing_skills(self, resume_skills: List[str], jd_data: Dict) -> List[str]:
        """Identify missing skills from job requirements"""
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        
        missing_skills = []
        
        # Check must-have skills
        for skill in jd_data['must_have_skills']:
            skill_lower = skill.lower()
            # Use fuzzy matching to avoid false negatives
            best_match = process.extractOne(skill_lower, resume_skills_lower)
            if not best_match or best_match[1] < 70:  # Lower threshold for missing skills
                missing_skills.append(skill)
        
        return missing_skills
    
    def generate_suggestions(self, missing_skills: List[str], jd_data: Dict) -> List[str]:
        """Generate improvement suggestions for candidates"""
        suggestions = []
        
        if missing_skills:
            suggestions.append(f"Consider gaining experience in: {', '.join(missing_skills[:5])}")
        
        if jd_data['qualifications']:
            suggestions.append(f"Ensure your education aligns with requirements: {', '.join(jd_data['qualifications'][:2])}")
        
        suggestions.append("Add more relevant projects showcasing your technical skills")
        suggestions.append("Include certifications related to the job requirements")
        suggestions.append("Quantify your achievements with specific metrics and results")
        
        return suggestions

class DatabaseManager:
    """Handles database operations for storing results"""
    
    def __init__(self, db_path: str = "resume_analysis.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
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
        """Get all analysis results from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM resume_analysis ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            result = {
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
            }
            results.append(result)
        
        conn.close()
        return results
    
    def filter_results(self, job_role: str = None, min_score: float = None, verdict: str = None) -> List[Dict]:
        """Filter analysis results based on criteria"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM resume_analysis WHERE 1=1"
        params = []
        
        if job_role:
            query += " AND job_role LIKE ?"
            params.append(f"%{job_role}%")
        
        if min_score is not None:
            query += " AND relevance_score >= ?"
            params.append(min_score)
        
        if verdict:
            query += " AND verdict = ?"
            params.append(verdict)
        
        query += " ORDER BY relevance_score DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            result = {
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
            }
            results.append(result)
        
        conn.close()
        return results

# Main application class
class ResumeRelevanceSystem:
    """Main system class that orchestrates all components"""
    
    def __init__(self):
        self.analyzer = RelevanceAnalyzer()
        self.db_manager = DatabaseManager()
    
    def process_resume(self, resume_file, jd_text: str, resume_filename: str = "") -> Dict[str, Any]:
        """Process a single resume against job description"""
        
        # Extract text from resume
        if resume_filename.lower().endswith('.pdf'):
            resume_text = self.analyzer.resume_parser.extract_text_from_pdf(resume_file)
        elif resume_filename.lower().endswith('.docx'):
            resume_text = self.analyzer.resume_parser.extract_text_from_docx(resume_file)
        else:
            # Try to read as text
            if hasattr(resume_file, 'read'):
                resume_text = resume_file.read().decode('utf-8', errors='ignore')
            else:
                resume_text = str(resume_file)
        
        # Clean text
        resume_text = self.analyzer.resume_parser.clean_and_normalize_text(resume_text)
        
        # Analyze resume
        result = self.analyzer.analyze_resume(resume_text, jd_text, resume_filename)
        
        # Save to database
        result_id = self.db_manager.save_analysis(result)
        result['id'] = result_id
        
        return result
    
    def process_multiple_resumes(self, resume_files: List, jd_text: str) -> List[Dict[str, Any]]:
        """Process multiple resumes against job description"""
        results = []
        
        for i, resume_file in enumerate(resume_files):
            filename = f"Resume_{i+1}"
            if hasattr(resume_file, 'name'):
                filename = resume_file.name
            
            try:
                result = self.process_resume(resume_file, jd_text, filename)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                # Add error result
                error_result = {
                    'resume_filename': filename,
                    'relevance_score': 0,
                    'verdict': 'Error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(error_result)
        
        return results
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display"""
        all_results = self.db_manager.get_all_results()
        
        if not all_results:
            return {
                'total_resumes': 0,
                'high_matches': 0,
                'medium_matches': 0,
                'low_matches': 0,
                'avg_score': 0,
                'results': []
            }
        
        # Calculate statistics
        high_matches = len([r for r in all_results if r['verdict'] == 'High'])
        medium_matches = len([r for r in all_results if r['verdict'] == 'Medium'])
        low_matches = len([r for r in all_results if r['verdict'] == 'Low'])
        avg_score = np.mean([r['relevance_score'] for r in all_results])
        
        return {
            'total_resumes': len(all_results),
            'high_matches': high_matches,
            'medium_matches': medium_matches,
            'low_matches': low_matches,
            'avg_score': round(avg_score, 2),
            'results': all_results
        }

if __name__ == "__main__":
    # Initialize system
    system = ResumeRelevanceSystem()
    
    # Example usage
    sample_jd = """
    We are looking for a Data Analyst with experience in Python, SQL, and data visualization.
    
    Required Skills:
    - Python (Pandas, NumPy)
    - SQL (MySQL, PostgreSQL)
    - Data visualization tools (Power BI, Tableau)
    - Statistics and data analysis
    
    Qualifications:
    - Bachelor's degree in relevant field
    - 2+ years of experience
    """
    
    print("Resume Relevance Check System initialized successfully!")
    print("Use the Streamlit web interface to upload resumes and job descriptions.")
