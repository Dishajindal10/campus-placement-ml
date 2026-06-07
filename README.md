# 🎓 Student Placement Prediction

A machine learning web app that predicts whether a student will get placed based on their academic profile.

## 🔍 Problem Statement

Campus placements depend on multiple factors — CGPA, internships, projects, aptitude, and more. This project builds a classification model to predict placement outcomes and help students identify areas for improvement.

## 📊 Dataset

Synthetic dataset of 500 students with the following features:

| Feature | Description |
|---|---|
| CGPA | Academic performance (5.0 – 10.0) |
| Internships | Number of internships completed |
| Projects | Number of projects built |
| Aptitude Score | Score out of 100 |
| Soft Skills | Rating out of 5 |
| Backlogs | Number of backlogs |
| Placed | Target variable (0 = Not Placed, 1 = Placed) |

## 🤖 Models Used

| Model | Accuracy |
|---|---|
| Logistic Regression | 92% |
| Random Forest | 89% |


## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/student-placement-prediction.git
cd student-placement-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset
python generate_data.py

# 4. Train the model
python train_model.py

# 5. Run the Streamlit app
streamlit run app.py
```

## 📈 Key Insights

- **CGPA** is the strongest predictor of placement
- Students with **1+ internships** have significantly higher placement rates
- **Backlogs** negatively impact placement chances
- **Aptitude score** above 60 correlates with better placement outcomes

## 🛠️ Tech Stack

- **Python** — Core language
- **Pandas & NumPy** — Data manipulation
- **Scikit-learn** — ML models
- **Matplotlib & Seaborn** — Visualizations
- **Streamlit** — Web app deployment

## Limitations

- Dataset is synthetically generated.
- Real-world placement outcomes depend on additional factors.
- Future work includes training on real placement datasets and deploying the application publicly.

## 👩‍💻 Author

built by **Disha Jindal**
