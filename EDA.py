import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
from fpdf import FPDF
np.random.seed(42)
departments = ['Sales', 'Technical', 'HR', 'Finance', 'R&D', 'Support', 'Management']
salary_bands = ['Low', 'Medium', 'High']
num_records = 1000
data = {
    'EmployeeID': [i for i in range(1, num_records + 1)],
    'Age': np.random.randint(22, 60, num_records),
    'Department': np.random.choice(departments, num_records),
    'SalaryBand': np.random.choice(salary_bands, num_records, p=[0.5, 0.3, 0.2]),
    'YearsAtCompany': np.random.randint(0, 20, num_records),
    'YearsSinceLastPromotion': np.random.randint(0, 10, num_records),
    'NumProjects': np.random.randint(1, 10, num_records),
    'SatisfactionLevel': np.round(np.random.uniform(0.2, 1.0, num_records), 2),
    'WorkLifeBalance': np.random.randint(1, 5, num_records),
    'Left': np.random.choice([0, 1], num_records, p=[0.75, 0.25])
}
df = pd.DataFrame(data)
df.to_csv("hr_attrition_dataset.csv", index=False)
print("\n--- EDA: Attrition Count by Department ---")
sns.countplot(data=df, x='Department', hue='Left')
plt.title("Attrition by Department")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("department_attrition.png")
plt.clf()
X = df.drop(['EmployeeID', 'Left'], axis=1)
X = pd.get_dummies(X)  
X = X.astype(np.float32)  
y = df['Left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png")
plt.clf()
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix_str = "\nConfusion Matrix:\n"
conf_matrix_str += f"{conf_matrix[0][0]:>5} {conf_matrix[0][1]:>5}\n"
conf_matrix_str += f"{conf_matrix[1][0]:>5} {conf_matrix[1][1]:>5}\n"
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", "B", 14)
pdf.cell(0, 10, "Model Accuracy Report", ln=True, align="C")
pdf.ln(10)
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "Confusion Matrix", ln=True)
pdf.set_font("Arial", "", 12)
pdf.multi_cell(0, 10, conf_matrix_str)
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "Classification Report", ln=True)
pdf.set_font("Arial", "", 12)
pdf.multi_cell(0, 10, class_report)
pdf.output("Model_Accuracy_Report.pdf")
print("PDF report saved as 'Model_Accuracy_Report.pdf'")