from flask import Flask, render_template, request
import os
import re
import joblib
import PyPDF2


app = Flask(__name__)

# ==========================================
# Paths
# ==========================================

MODEL_PATH = os.path.join("models", "resume_model.pkl")
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==========================================
# Load AI Model
# ==========================================

try:
    model = joblib.load(MODEL_PATH)
    print("AI Resume Model Loaded Successfully!")
except Exception as e:
    model = None
    print("Error loading model:", e)


# ==========================================
# Extract text from PDF
# ==========================================

def extract_text_from_pdf(pdf_path):

    text = ""

    try:
        with open(pdf_path, "rb") as pdf_file:

            reader = PyPDF2.PdfReader(pdf_file)

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

    except Exception as e:
        print("PDF extraction error:", e)

    return text


# ==========================================
# Extract Skills
# ==========================================

def extract_skills(text):

    skills_list = [
        "Python",
        "Java",
        "C",
        "C++",
        "SQL",
        "HTML",
        "CSS",
        "JavaScript",
        "React",
        "Node.js",
        "Flask",
        "Django",
        "Machine Learning",
        "Deep Learning",
        "Artificial Intelligence",
        "Data Science",
        "Data Analytics",
        "Pandas",
        "NumPy",
        "Matplotlib",
        "Scikit-learn",
        "TensorFlow",
        "PyTorch",
        "Power BI",
        "Tableau",
        "Excel",
        "Git",
        "GitHub",
        "AWS",
        "Azure",
        "MySQL",
        "MongoDB"
    ]

    found_skills = []

    text_lower = text.lower()

    for skill in skills_list:

        if skill.lower() in text_lower:
            found_skills.append(skill)

    return found_skills


# ==========================================
# Career Recommendations
# ==========================================

def get_career_recommendations(category, skills):

    category_lower = category.lower()

    recommendations = []

    if "data" in category_lower:
        recommendations.extend([
            "Data Analyst",
            "Data Scientist",
            "Business Analyst",
            "Data Engineer"
        ])

    elif "python" in category_lower:
        recommendations.extend([
            "Python Developer",
            "Backend Developer",
            "Software Developer"
        ])

    elif "java" in category_lower:
        recommendations.extend([
            "Java Developer",
            "Backend Developer",
            "Software Developer"
        ])

    elif "web" in category_lower:
        recommendations.extend([
            "Web Developer",
            "Frontend Developer",
            "Full Stack Developer"
        ])

    elif "machine" in category_lower or "artificial" in category_lower:
        recommendations.extend([
            "Machine Learning Engineer",
            "AI Engineer",
            "Data Scientist"
        ])

    elif "hr" in category_lower:
        recommendations.extend([
            "HR Executive",
            "HR Analyst",
            "Recruitment Specialist"
        ])

    elif "finance" in category_lower:
        recommendations.extend([
            "Financial Analyst",
            "Finance Executive",
            "Business Analyst"
        ])

    else:
        recommendations.extend([
            "Software Developer",
            "Data Analyst",
            "Business Analyst",
            "IT Support Engineer"
        ])

    # Add skill-based recommendations

    if "Machine Learning" in skills or "Deep Learning" in skills:
        if "Machine Learning Engineer" not in recommendations:
            recommendations.append("Machine Learning Engineer")

    if "Power BI" in skills or "Tableau" in skills:
        if "Data Analyst" not in recommendations:
            recommendations.append("Data Analyst")

    if "React" in skills or "JavaScript" in skills:
        if "Frontend Developer" not in recommendations:
            recommendations.append("Frontend Developer")

    if "Flask" in skills or "Django" in skills:
        if "Backend Developer" not in recommendations:
            recommendations.append("Backend Developer")

    return list(dict.fromkeys(recommendations))[:6]


# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# Resume Analysis
# ==========================================

@app.route("/analyze", methods=["POST"])
def analyze():

    if model is None:

        return """
        <h2>AI Model not found.</h2>
        <p>Please run train_model.py first.</p>
        """

    if "resume" not in request.files:

        return "No resume file uploaded."

    file = request.files["resume"]

    if file.filename == "":

        return "Please select a resume PDF."

    # Check PDF
    if not file.filename.lower().endswith(".pdf"):

        return "Please upload a PDF resume."

    # Save uploaded file
    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(file_path)

    # Extract resume text
    resume_text = extract_text_from_pdf(file_path)

    if not resume_text.strip():

        return """
        <h2>Could not extract text from the PDF.</h2>
        <p>Please upload a text-based PDF resume.</p>
        """

    # ======================================
    # AI Prediction
    # ======================================

    prediction = model.predict([resume_text])

    category = prediction[0]

    # ======================================
    # Prediction Probability
    # ======================================

    confidence = None

    try:

        probabilities = model.predict_proba([resume_text])

        confidence = round(
            max(probabilities[0]) * 100,
            2
        )

    except Exception:

        confidence = None

    # ======================================
    # Skill Extraction
    # ======================================

    skills = extract_skills(resume_text)

    # ======================================
    # Career Recommendations
    # ======================================

    careers = get_career_recommendations(
        category,
        skills
    )

    # ======================================
    # Resume Statistics
    # ======================================

    words = resume_text.split()

    word_count = len(words)

    # ======================================
    # Render Result
    # ======================================

    return render_template(
        "result.html",
        category=category,
        confidence=confidence,
        skills=skills,
        careers=careers,
        word_count=word_count
    )


# ==========================================
# Run Flask Application
# ==========================================

if __name__ == "__main__":

    print("-------------------------------------")
    print("AI Resume Analyzer")
    print("-------------------------------------")
    print("Server starting...")
    print("Open: http://127.0.0.1:5000")
    print("-------------------------------------")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
