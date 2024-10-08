import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv('/content/Student_performance_data _.csv')
df.drop('StudentID',axis=1,inplace=True)
corr=df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar_kws={"shrink": .8})

plt.title('Correlation Matrix Heatmap (Lower Triangle)', fontsize=20)

plt.show()

colors = sns.color_palette('pastel')

def plot_pie(ax, data, column, labels, title):
    counts = data[column].value_counts().sort_index()
    sizes = counts.values
    ax.pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%', startangle=140, pctdistance=0.85, explode=[0.1] + [0]*(len(labels)-1))
    ax.set_title(title, fontsize=14)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

plot_pie(axes[0], df, 'GradeClass', ['A', 'B', 'C', 'D', 'F'], 'Grade Distribution')
plot_pie(axes[1], df, 'Gender', ['Male', 'Female'], 'Gender Distribution')
plot_pie(axes[2], df, 'Ethnicity', ['Caucasian', 'African American', 'Asian', 'Other'], 'Ethnicity Distribution')

plt.tight_layout()

plt.show()

x=df.drop('GradeClass',axis=1)
y=df['GradeClass']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,stratify=y,shuffle=True,random_state=15)

classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

for name, clf in classifiers.items():
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name}: {accuracy:.2f}')
    print('**'*10)

random_forest_clf=RandomForestClassifier(n_estimators=200,random_state=15,min_samples_split=5,min_samples_leaf=4,max_features=None)
random_forest_clf.fit(x_train, y_train)
y_pred = random_forest_clf.predict(x_test)
accuracy_score(y_test, y_pred)
