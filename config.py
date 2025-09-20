"""
Configuration file for Resume Relevance Check System
Modify these settings to customize the application behavior
"""

# Application Settings
APP_CONFIG = {
    'title': 'Resume Relevance Check System',
    'subtitle': 'AI-powered resume evaluation against job requirements | Innomatics Research Labs',
    'version': '1.0.0',
    'debug_mode': False
}

# Scoring Configuration
SCORING_CONFIG = {
    # Final score weights
    'final_weights': {
        'hard_match': 0.7,      # Keyword/skill matching
        'semantic_match': 0.3   # Contextual similarity
    },
    
    # Hard match component weights
    'hard_match_weights': {
        'skills': 0.7,          # Skill matching importance
        'education': 0.3        # Education matching importance
    },
    
    # Verdict thresholds
    'verdict_thresholds': {
        'high': 75,             # High suitability threshold
        'medium': 50,           # Medium suitability threshold
        'low': 0               # Low suitability threshold
    }
}

# Text Processing Settings
TEXT_PROCESSING = {
    'max_file_size_mb': 10,
    'supported_formats': ['pdf', 'txt'],
    'min_text_length': 50,
    'max_text_length': 50000
}

# Skill Keywords Database
SKILL_KEYWORDS = {
    'programming_languages': [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby',
        'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl', 'c'
    ],
    
    'data_science': [
        'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
        'matplotlib', 'seaborn', 'plotly', 'jupyter', 'anaconda', 'statsmodels',
        'scipy', 'nltk', 'spacy', 'opencv', 'machine learning', 'deep learning',
        'data analysis', 'data visualization', 'statistics', 'eda'
    ],
    
    'web_technologies': [
        'html', 'css', 'react', 'angular', 'vue.js', 'node.js', 'express.js',
        'django', 'flask', 'fastapi', 'spring boot', 'bootstrap', 'tailwind css',
        'jquery', 'webpack', 'babel', 'sass', 'less'
    ],
    
    'databases': [
        'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
        'sqlite', 'oracle', 'sql server', 'dynamodb', 'neo4j', 'sql'
    ],
    
    'cloud_devops': [
        'aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'terraform',
        'ansible', 'jenkins', 'gitlab ci', 'github actions', 'circleci'
    ],
    
    'analytics_tools': [
        'power bi', 'tableau', 'excel', 'google analytics', 'looker', 'qlik',
        'sas', 'spss', 'alteryx', 'd3.js', 'databricks', 'spark', 'kafka'
    ],
    
    'soft_skills': [
        'communication', 'leadership', 'teamwork', 'problem solving',
        'analytical thinking', 'project management', 'time management',
        'adaptability', 'creativity', 'attention to detail', 'collaboration'
    ]
}

# Education Keywords
EDUCATION_KEYWORDS = {
    'bachelors': [
        'bachelor', 'b.tech', 'btech', 'b.e', 'be', 'bachelor of technology',
        'bachelor of engineering', 'b.sc', 'bsc', 'bachelor of science',
        'b.com', 'bcom', 'bachelor of commerce', 'b.ca', 'bca',
        'bachelor of computer application', 'graduation', 'graduate'
    ],
    
    'masters': [
        'master', 'm.tech', 'mtech', 'm.e', 'me', 'master of technology',
        'master of engineering', 'm.sc', 'msc', 'master of science',
        'mba', 'master of business administration', 'm.com', 'mcom',
        'master of commerce', 'm.ca', 'mca', 'master of computer application',
        'post graduation', 'postgraduate'
    ],
    
    'doctorate': [
        'phd', 'ph.d', 'doctorate', 'doctor of philosophy'
    ]
}

# Sample Job Descriptions for Quick Testing
SAMPLE_JDS = {
    'Data Science Role': """We are looking for a Data Analyst with experience in Python, SQL, and data visualization.

Required Skills:
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- SQL (MySQL, PostgreSQL) 
- Data visualization tools (Power BI, Tableau)
- Statistics and data analysis
- Machine learning basics
- Excel proficiency

Qualifications:
- Bachelor's degree in Engineering, Statistics, or related field
- 2+ years of experience in data analysis
- Strong analytical and problem-solving skills

Experience:
- Experience with EDA and statistical analysis
- Knowledge of data cleaning and preprocessing
- Familiarity with business intelligence tools""",

    'Software Engineering Role': """We are seeking a Software Engineer to join our development team.

Required Skills:
- Python, Java, or JavaScript programming
- Web development frameworks (React, Angular, Django, Flask)
- Database knowledge (MySQL, PostgreSQL, MongoDB)
- Git version control
- API development experience
- HTML, CSS, JavaScript

Qualifications:
- Bachelor's degree in Computer Science or related field
- 1+ years of software development experience
- Strong problem-solving abilities

Experience:
- Full-stack development experience
- Agile development methodology
- Experience with cloud platforms (AWS, Azure)""",

    'Business Analyst Role': """Looking for a Business Analyst to drive data-driven decision making.

Required Skills:
- SQL for data extraction and analysis
- Excel advanced functions and pivot tables
- Power BI or Tableau for visualization
- Business intelligence tools
- Data analysis and interpretation
- Statistical analysis

Qualifications:
- Bachelor's degree in Business, Engineering, or related field
- 2+ years of business analysis experience
- Strong communication skills

Experience:
- Requirements gathering and documentation
- Process improvement and optimization
- Stakeholder management"""
}

# UI Customization
UI_CONFIG = {
    'theme_colors': {
        'primary': '#667eea',
        'secondary': '#764ba2', 
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'info': '#17a2b8'
    },
    
    'verdict_colors': {
        'High': '#28a745',
        'Medium': '#ffc107', 
        'Low': '#dc3545'
    }
}

# Database Configuration
DATABASE_CONFIG = {
    'name': 'resume_analysis.db',
    'backup_enabled': True,
    'backup_frequency': 'daily'
}

# Function to get all skills as a flat list
def get_all_skills():
    """Return all skills from all categories as a single list"""
    all_skills = []
    for category_skills in SKILL_KEYWORDS.values():
        all_skills.extend(category_skills)
    return sorted(list(set(all_skills)))

# Function to get skills by category
def get_skills_by_category(category):
    """Return skills for a specific category"""
    return SKILL_KEYWORDS.get(category, [])

# Function to get all education keywords
def get_all_education_keywords():
    """Return all education keywords as a single list"""
    all_education = []
    for level_keywords in EDUCATION_KEYWORDS.values():
        all_education.extend(level_keywords)
    return sorted(list(set(all_education)))

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Total skills in database: {len(get_all_skills())}")
    print(f"Total education keywords: {len(get_all_education_keywords())}")