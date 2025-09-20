"""
Configuration file for Resume Relevance Check System
"""

import os
from typing import Dict, List

# System Configuration
SYSTEM_CONFIG = {
    'app_name': 'Resume Relevance Check System',
    'version': '1.0.0',
    'debug_mode': os.getenv('DEBUG', 'False').lower() == 'true',
    'database_path': 'resume_analysis.db'
}

# Scoring Configuration
SCORING_CONFIG = {
    'weights': {
        'hard_match': 0.7,
        'semantic_match': 0.3
    },
    'hard_match_weights': {
        'must_have_skills': 0.4,
        'good_to_have_skills': 0.2,
        'qualifications': 0.3,
        'general_match': 0.1
    },
    'verdict_thresholds': {
        'high': 75,
        'medium': 50,
        'low': 0
    },
    'fuzzy_match_threshold': 80,
    'missing_skill_threshold': 70
}

# Text Processing Configuration
TEXT_CONFIG = {
    'max_text_length': 50000,
    'min_text_length': 100,
    'sentence_model': 'all-MiniLM-L6-v2',
    'tfidf_max_features': 1000,
    'tfidf_ngram_range': (1, 2)
}

# Skill Keywords Database
SKILL_KEYWORDS = {
    'programming_languages': [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 
        'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl'
    ],
    'web_technologies': [
        'html', 'css', 'react', 'angular', 'vue.js', 'node.js', 'express.js',
        'django', 'flask', 'fastapi', 'spring boot', 'bootstrap', 'tailwind css',
        'jquery', 'webpack', 'babel'
    ],
    'databases': [
        'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
        'sqlite', 'oracle', 'sql server', 'dynamodb', 'neo4j'
    ],
    'data_science': [
        'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
        'matplotlib', 'seaborn', 'plotly', 'jupyter', 'anaconda', 'statsmodels',
        'scipy', 'nltk', 'spacy', 'opencv'
    ],
    'analytics_tools': [
        'power bi', 'tableau', 'excel', 'google analytics', 'looker', 'qlik',
        'sas', 'spss', 'alteryx', 'd3.js'
    ],
    'cloud_platforms': [
        'aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'terraform',
        'ansible', 'jenkins', 'gitlab ci', 'github actions'
    ],
    'soft_skills': [
        'communication', 'leadership', 'teamwork', 'problem solving',
        'analytical thinking', 'project management', 'time management',
        'adaptability', 'creativity', 'attention to detail'
    ]
}

# Education Keywords
EDUCATION_KEYWORDS = {
    'bachelors': [
        'bachelor', 'b.tech', 'btech', 'b.e', 'be', 'bachelor of technology',
        'bachelor of engineering', 'b.sc', 'bsc', 'bachelor of science',
        'b.com', 'bcom', 'bachelor of commerce', 'b.ca', 'bca',
        'bachelor of computer application'
    ],
    'masters': [
        'master', 'm.tech', 'mtech', 'm.e', 'me', 'master of technology',
        'master of engineering', 'm.sc', 'msc', 'master of science',
        'mba', 'master of business administration', 'm.com', 'mcom',
        'master of commerce', 'm.ca', 'mca', 'master of computer application'
    ],
    'doctorate': [
        'phd', 'ph.d', 'doctorate', 'doctor of philosophy'
    ]
}

# Job Role Categories
JOB_ROLE_CATEGORIES = {
    'data_science': [
        'data scientist', 'data analyst', 'business analyst', 'research analyst',
        'machine learning engineer', 'ai engineer', 'statistician'
    ],
    'software_engineering': [
        'software engineer', 'software developer', 'full stack developer',
        'backend developer', 'frontend developer', 'web developer',
        'mobile developer', 'devops engineer'
    ],
    'management': [
        'product manager', 'project manager', 'team lead', 'technical lead',
        'engineering manager', 'data manager'
    ]
}

# File Upload Configuration
UPLOAD_CONFIG = {
    'max_file_size': 10 * 1024 * 1024,  # 10 MB
    'allowed_extensions': ['pdf', 'docx', 'txt'],
    'max_files_bulk': 50
}

# UI Configuration
UI_CONFIG = {
    'theme': {
        'primary_color': '#667eea',
        'secondary_color': '#764ba2',
        'success_color': '#28a745',
        'warning_color': '#ffc107',
        'danger_color': '#dc3545'
    },
    'page_config': {
        'page_title': 'Resume Relevance Check System',
        'page_icon': '📄',
        'layout': 'wide'
    }
}

def get_all_skills() -> List[str]:
    """Get all skills from the skill keywords database"""
    all_skills = []
    for category, skills in SKILL_KEYWORDS.items():
        all_skills.extend(skills)
    return list(set(all_skills))

def get_skills_by_category(category: str) -> List[str]:
    """Get skills for a specific category"""
    return SKILL_KEYWORDS.get(category, [])

def get_education_patterns() -> List[str]:
    """Get all education patterns for regex matching"""
    all_patterns = []
    for level, patterns in EDUCATION_KEYWORDS.items():
        all_patterns.extend(patterns)
    return all_patterns