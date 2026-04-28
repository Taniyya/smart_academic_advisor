from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import sqlite3
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
if os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_KEY") != "your_api_key_here":
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

ADVISOR_SYSTEM_PROMPT = """You are a highly advanced AI-powered Academic Advisor integrated into the Smart Academic Advisor system. Your persona is similar to ChatGPT—helpful, extremely intelligent, highly conversational, and warm.

Objective:
Provide world-class, deeply thoughtful guidance to students in their academic journey by analyzing their specific context, interests, performance, and challenges.

Instructions:
1. Conversational & Empathetic: Adopt a very warm, human-like, and mentoring tone. Speak directly to the student as if you are a highly supportive, knowledgeable professor.
2. In-Depth Analysis: Do not just give short answers. Provide highly detailed breakdowns of what the student's specific marks and attendance mean for their future.
3. Comprehensive Advice: If a student has a weak subject, give them a highly structured, step-by-step study plan including time-management strategies.
4. Professional Formatting: Use rich Markdown formatting! Use bold text for emphasis, bulleted lists for actionable steps, structured headers, and emojis to keep things engaging.
5. Be Thorough: Detail the *why* and *how* for all your suggestions, offering concrete real-world examples.

Goal: Deliver deeply insightful, comprehensive, and high-quality responses that provide immense value and clarity.
"""

app = Flask(__name__)

# Database setup
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (Student_ID TEXT PRIMARY KEY, Gender TEXT, "Attendance_%" REAL,
                  Internal_Marks REAL, Sem1 REAL, Sem2 REAL, Sem3 REAL,
                  Sem4 REAL, Sem5 REAL, Sem6 REAL, Average_Marks REAL,
                  CGPA REAL, Grade TEXT, Career_Suggestion TEXT)''')
    conn.commit()
    conn.close()

def load_data_to_db():
    df = pd.read_csv('data.csv', sep='\t')
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('DELETE FROM students')  # Clear existing data
    conn.commit()
    df.to_sql('students', conn, if_exists='append', index=False)
    conn.close()

# Load and preprocess data
def load_and_preprocess():
    df = pd.read_csv('data.csv', sep='\t')
    print("Columns:", df.columns.tolist())
    # For simplicity, assume no skills column, use existing features
    features = ['Attendance_%', 'Internal_Marks', 'Sem1', 'Sem2', 'Sem3', 'Sem4', 'Sem5', 'Sem6', 'Average_Marks', 'CGPA']
    target = 'Career_Suggestion'

    # Encode target
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])

    X = df[features]
    y = df[target]

    return X, y, le

# Train model
def train_model():
    X, y, le = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    svm = SVC(kernel='linear', probability=True, random_state=42)
    nb = GaussianNB()

    model = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('svm', svm), ('nb', nb)],
        voting='hard'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')

    joblib.dump(model, 'model/trained_model.pkl')
    joblib.dump(le, 'model/label_encoder.pkl')

    return accuracy

# Load model
def load_model():
    model = joblib.load('model/trained_model.pkl')
    le = joblib.load('model/label_encoder.pkl')
    return model, le

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    students = []
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT Student_ID FROM students')
    rows = c.fetchall()
    for row in rows:
        students.append({'id': row[0], 'name': row[0]})  # Using Student_ID as name
    conn.close()
    return render_template('dashboard.html', students=students)

@app.route('/get_student_data/<student_id>')
def get_student_data(student_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT * FROM students WHERE Student_ID = ?', (student_id,))
    row = c.fetchone()
    conn.close()

    if row:
        student = {
            'id': row[0],
            'gender': row[1],
            'attendance': row[2],
            'internal_marks': row[3],
            'sem1': row[4],
            'sem2': row[5],
            'sem3': row[6],
            'sem4': row[7],
            'sem5': row[8],
            'sem6': row[9],
            'average_marks': row[10],
            'cgpa': row[11],
            'grade': row[12],
            'career': row[13]
        }
        return jsonify(student)
    return jsonify({'error': 'Student not found'})

@app.route('/predict_career/<student_id>')
def predict_career(student_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT "Attendance_%", Internal_Marks, Sem1, Sem2, Sem3, Sem4, Sem5, Sem6, Average_Marks, CGPA FROM students WHERE Student_ID = ?', (student_id,))
    row = c.fetchone()
    conn.close()

    if row:
        features = np.array(row).reshape(1, -1)
        model, le = load_model()
        prediction = model.predict(features)
        career = le.inverse_transform(prediction)[0]
        return jsonify({'career': career})
    return jsonify({'error': 'Student not found'})

@app.route('/get_recommendations/<student_id>', methods=['POST'])
def get_recommendations(student_id):
    qa_answers = request.get_json()
    
    # Get ML prediction
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT "Attendance_%", Internal_Marks, Sem1, Sem2, Sem3, Sem4, Sem5, Sem6, Average_Marks, CGPA FROM students WHERE Student_ID = ?', (student_id,))
    row = c.fetchone()
    conn.close()

    career = 'IT Jobs / Private Sector'  # Default
    ai_breakdown = {}
    if row:
        features = np.array(row).reshape(1, -1)
        model, le = load_model()
        prediction = model.predict(features)
        base_career = le.inverse_transform(prediction)[0]
        career = base_career
        
        # Individual AI Breakdown
        try:
            if hasattr(model, 'estimators_'):
                ai_breakdown = {
                    "Random Forest": le.inverse_transform(model.estimators_[0].predict(features))[0],
                    "Gradient Boosting": le.inverse_transform(model.estimators_[1].predict(features))[0],
                    "Support Vector Machine (SVM)": le.inverse_transform(model.estimators_[2].predict(features))[0],
                    "Naive Bayes": le.inverse_transform(model.estimators_[3].predict(features))[0]
                }
        except:
            pass

    # Fusion Logic: Merge Base ML Prediction with 30 Standard Questions Mapping
    it_triggers = ['coding', 'logical', 'new_tech', 'private', 'real_world', 'tools_comfort', 'debugging', 'upgrade_skills', 'it_career', 'startup', 'complex_decisions', 'responsibility', 'interacting', 'presenting', 'leading', 'initiative', 'abroad', 'own_business']
    if any(qa_answers.get(key) == 'yes' for key in it_triggers):
        career = 'IT Jobs / Private Sector'
    elif qa_answers.get('research') == 'yes' or qa_answers.get('theoretical') == 'yes' or qa_answers.get('hands_on') == 'no':
        career = 'Higher Studies / Research'
    elif qa_answers.get('stable') == 'stable' or qa_answers.get('government') == 'yes' or qa_answers.get('pressure') == 'no' or qa_answers.get('work_env') == 'structured':
        career = 'Government / Public Sector'
    elif qa_answers.get('team') == 'team' or qa_answers.get('logical') == 'no' or qa_answers.get('multitasking') == 'no':
        career = 'Skill Development Courses'

    # Handle Behavioral Overrides
    if career == 'IT Jobs / Private Sector' and (qa_answers.get('theoretical') == 'yes' or qa_answers.get('creativity') == 'yes'):
        career = 'Higher Studies / Research'
    if career == 'Government / Public Sector' and qa_answers.get('risk') == 'yes':
        career = 'IT Jobs / Private Sector'

    # Skills and improvements
    skills = ['Python', 'Data Analysis', 'Communication']
    improvements = ['Focus on weak subjects', 'Improve attendance', 'Practice coding daily']

    if career == 'Higher Studies / Research':
        skills = ['Research Methodology', 'Advanced Mathematics', 'Writing Skills']
        improvements = ['Publish papers', 'Attend conferences', 'Collaborate on projects']
    elif career == 'Skill Development Courses':
        skills = ['Basic Programming', 'Soft Skills', 'Time Management']
        improvements = ['Enroll in courses', 'Build portfolio', 'Network with professionals']

    return jsonify({
        'career': career,
        'skills': skills,
        'improvements': improvements,
        'ai_breakdown': ai_breakdown
    })

@app.route('/qa/<student_id>')
def qa(student_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT Sem1, Sem2, Sem3, Sem4, Sem5, Sem6, Student_ID FROM students WHERE Student_ID = ?', (student_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return "Student not found", 404
    
    s1, s2, s3, s4, s5, s6, student_id = row
    sgpas = [x for x in [s1, s2, s3, s4, s5, s6] if x > 0]
    avg_sgpa = sum(sgpas) / len(sgpas) if sgpas else 0
    
    # Dynamic Questionnaire generation
    questions = []
    
    # Questions Dictionary Base
    q_dict = {
        1: {"id": "coding", "label": "Do you enjoy coding and programming?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        2: {"id": "logical", "label": "Do you like solving logical or analytical problems?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        3: {"id": "new_tech", "label": "Are you interested in learning new technologies?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        4: {"id": "research", "label": "Are you interested in research or higher studies?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        5: {"id": "theoretical", "label": "Do you enjoy theoretical subjects more than practical work?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        6: {"id": "stable", "label": "Do you prefer a stable job over a high-paying risky job?", "options": [{"value": "stable", "label": "Stable"}, {"value": "high", "label": "High-Paying/Risky"}]},
        7: {"id": "government", "label": "Are you interested in government jobs?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        8: {"id": "private", "label": "Do you prefer working in a corporate/private company?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        9: {"id": "real_world", "label": "Do you enjoy working on real-world projects?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        10: {"id": "team", "label": "Do you prefer working independently or in a team?", "options": [{"value": "independent", "label": "Independently"}, {"value": "team", "label": "In a Team"}]},
        11: {"id": "data_numbers", "label": "Do you enjoy working with data and numbers?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        12: {"id": "tools_comfort", "label": "Are you comfortable using tools like Excel, Python, or SQL?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        13: {"id": "debugging", "label": "Do you like debugging and fixing errors in code?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        14: {"id": "hands_on", "label": "Do you prefer hands-on learning over theoretical learning?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        15: {"id": "upgrade_skills", "label": "Are you willing to continuously upgrade your technical skills?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        16: {"id": "startup", "label": "Are you interested in startup environments?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        17: {"id": "creativity", "label": "Do you prefer a job with creativity and innovation?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        18: {"id": "it_career", "label": "Do you want to pursue a career in the IT/Tech industry?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        19: {"id": "risk", "label": "Are you comfortable taking risks in your career?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        20: {"id": "pressure", "label": "Do you enjoy working under pressure and deadlines?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        21: {"id": "complex_decisions", "label": "Do you enjoy making decisions in complex situations?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        22: {"id": "responsibility", "label": "Are you confident in taking responsibility for your work?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        23: {"id": "interacting", "label": "Do you like interacting and communicating with people?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        24: {"id": "presenting", "label": "Are you comfortable presenting your ideas in front of others?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        25: {"id": "leading", "label": "Do you like leading a team or managing projects?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        26: {"id": "initiative", "label": "Do you take initiative without being told?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        27: {"id": "work_env", "label": "Do you prefer structured tasks or flexible work?", "options": [{"value": "structured", "label": "Structured"}, {"value": "flexible", "label": "Flexible"}]},
        28: {"id": "multitasking", "label": "Do you enjoy multitasking?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        29: {"id": "abroad", "label": "Are you interested in working abroad?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]},
        30: {"id": "own_business", "label": "Do you want to start your own business in the future?", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}]}
    }

    # 1. 10 Completely Unique Student Demo Profiles mapping first batch Qs (1-10)
    if student_id == 'S1':
        questions.extend([q_dict[1], q_dict[2], q_dict[3], q_dict[4], q_dict[5]])
    elif student_id == 'S2':
        questions.extend([q_dict[6], q_dict[7], q_dict[8], q_dict[9], q_dict[10]])
    elif student_id == 'S3':
        questions.extend([q_dict[1], q_dict[3], q_dict[5], q_dict[7], q_dict[9]])
    elif student_id == 'S4':
        questions.extend([q_dict[2], q_dict[4], q_dict[6], q_dict[8], q_dict[10]])
    elif student_id == 'S5':
        questions.extend([q_dict[1], q_dict[2], q_dict[8], q_dict[9], q_dict[10]])
    elif student_id == 'S6':
        questions.extend([q_dict[3], q_dict[4], q_dict[5], q_dict[6], q_dict[7]])
    elif student_id == 'S7':
        questions.extend([q_dict[2], q_dict[3], q_dict[6], q_dict[7], q_dict[8]])
    elif student_id == 'S8':
        questions.extend([q_dict[1], q_dict[4], q_dict[5], q_dict[9], q_dict[10]])
    elif student_id == 'S9':
        questions.extend([q_dict[1], q_dict[5], q_dict[6], q_dict[7], q_dict[8]])
    elif student_id == 'S10':
        questions.extend([q_dict[2], q_dict[3], q_dict[4], q_dict[9], q_dict[10]])

    # 10 Unique Student Profiles mapping second batch Qs (11-20)
    elif student_id == 'S11':
        questions.extend([q_dict[11], q_dict[12], q_dict[13], q_dict[14], q_dict[15]])
    elif student_id == 'S12':
        questions.extend([q_dict[16], q_dict[17], q_dict[18], q_dict[19], q_dict[20]])
    elif student_id == 'S13':
        questions.extend([q_dict[11], q_dict[13], q_dict[15], q_dict[17], q_dict[19]])
    elif student_id == 'S14':
        questions.extend([q_dict[12], q_dict[14], q_dict[16], q_dict[18], q_dict[20]])
    elif student_id == 'S15':
        questions.extend([q_dict[11], q_dict[12], q_dict[18], q_dict[19], q_dict[20]])
    elif student_id == 'S16':
        questions.extend([q_dict[13], q_dict[14], q_dict[15], q_dict[16], q_dict[17]])
    elif student_id == 'S17':
        questions.extend([q_dict[12], q_dict[13], q_dict[16], q_dict[17], q_dict[18]])
    elif student_id == 'S18':
        questions.extend([q_dict[11], q_dict[14], q_dict[15], q_dict[19], q_dict[20]])
    elif student_id == 'S19':
        questions.extend([q_dict[11], q_dict[15], q_dict[16], q_dict[17], q_dict[18]])
    elif student_id == 'S20':
        questions.extend([q_dict[12], q_dict[13], q_dict[14], q_dict[19], q_dict[20]])

    else:
        # Fallback Dynamic Math-based SGPA context exactly 5 questions
        if avg_sgpa >= 7.3:
            questions.extend([q_dict[4], q_dict[16], q_dict[17], q_dict[18], q_dict[19]])
        elif avg_sgpa < 6.8:
            questions.extend([q_dict[14], q_dict[13], q_dict[5], q_dict[6], q_dict[20]])
        else:
            questions.extend([q_dict[1], q_dict[2], q_dict[12], q_dict[9], q_dict[8]])

    return render_template('qa.html', student_id=student_id, questions=questions, name=student_id)

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/api/analytics_data')
def analytics_data():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    # Fetch Career Suggestion distribution
    c.execute('SELECT Career_Suggestion, COUNT(*) FROM students GROUP BY Career_Suggestion')
    career_rows = c.fetchall()
    
    # Career Distribution (Pie Chart)
    career_labels = [row[0] for row in career_rows]
    career_values = [row[1] for row in career_rows]

    # CGPA Distribution per Career (Bar Chart)
    c.execute('SELECT Career_Suggestion, AVG(CGPA) FROM students GROUP BY Career_Suggestion')
    cgpa_rows = c.fetchall()
    cgpa_labels = [row[0] for row in cgpa_rows]
    cgpa_values = [round(row[1], 2) for row in cgpa_rows]

    conn.close()

    return jsonify({
        "careers": {"labels": career_labels, "values": career_values},
        "cgpa": {"labels": cgpa_labels, "values": cgpa_values}
    })

@app.route('/advisor/<student_id>')
def advisor(student_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT Student_ID FROM students WHERE Student_ID = ?', (student_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        return "Student not found", 404
        
    return render_template('chat.html', student_id=student_id)

@app.route('/api/chat/<student_id>', methods=['POST'])
def chat_api(student_id):
    data = request.get_json()
    history = data.get('history', [])
    message = data.get('message', '')

    if not message:
        return jsonify({"error": "No message provided"}), 400

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT Gender, "Attendance_%", Internal_Marks, Sem1, Sem2, Sem3, Sem4, Sem5, Sem6, Average_Marks, CGPA, Grade, Career_Suggestion FROM students WHERE Student_ID = ?', (student_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Student not found"}), 404

    student_context = f"""
--- CURRENT STUDENT CONTEXT ---
Student ID: {student_id}
Gender: {row[0]}
Attendance: {row[1]}%
Internal Marks: {row[2]}
Sem 1 to 6 Marks: {row[3]}, {row[4]}, {row[5]}, {row[6]}, {row[7]}, {row[8]}
Average Marks: {row[9]:.2f}
CGPA: {row[10]}
Grade: {row[11]}
Predicted Career Path: {row[12]}
-------------------------------
Please use this specific performance data to inform your advice.
"""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        return jsonify({"response": "Oops! The Gemini API Key has not been configured in the `.env` file yet. Please set it so I can assist you!"})

    try:
        model = genai.GenerativeModel(
            model_name='gemini-flash-latest',
            system_instruction=ADVISOR_SYSTEM_PROMPT + student_context
        )
        
        formatted_history = []
        for msg in history:
            role = "user" if msg["sender"] == "user" else "model"
            formatted_history.append({"role": role, "parts": [msg["text"]]})

        chat = model.start_chat(history=formatted_history)
        response = chat.send_message(message)

        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_db()
    load_data_to_db()
    if not os.path.exists('model/trained_model.pkl'):
        train_model()
    
    # Automatically open the browser
    import threading
    import webbrowser
    threading.Timer(1.25, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    
    app.run(debug=True, use_reloader=False)