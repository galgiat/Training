# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import plotly.express as px

from pandas.plotting import scatter_matrix
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf

# בקשה מפנדס להציג מספרים שכתובים בצורה מדעית בצורה קריאה ונוחה יותר
pd.options.display.float_format = "{:,.2f}".format
#   קראו את מערך הנתונים מקובץ CSV
df = pd.read_csv("DB/realtor-data1.csv", encoding='latin1')
#הציגו את השורות הראשונות של מערך הנתוני ם ואינפורמציה עליהם.
print("Top 5 Rows: ")
print(df.head(5),"\n")
#הדפיסו מידע על מערך הנתונים, כגון שמות עמודות, סוגי נתונים וערכים חסרים.
print("info: ")
df.info()
print("\n")
print("missing values:")
print(df.isnull().sum(),"\n")
#יש להדפיס סטטיסטיקה תיאורית של הנתונים ב - data-frame. חשוב להדפיס סטטיסטיקה תיאורית של משתנים רציפים ומשתנים קטגוריים כאחד.
print("data frame statics (numeric):")
print(df.describe(),"\n")
# סטטיסטיקה תיאורית למשתנים הקטגוריים (טקסטואליים) - count, unique, top, freq
print("data frame statics (categorical):")
print(df.describe(include='object'),"\n")

# שלב 3
#הסירו כפילויות ממערך הנתונים.
dfLen =len(df)
df = df.drop_duplicates()
print( f" {dfLen - len(df)} rows was deleted" , "\n")
#•  טפלו בערכים חסרים על ידי שחרור השורות או מילוין בערכים מתאימים.
# מחיקת שורות שהמחיר , העיר , או המדינה ריקה בהן מכיוון שאלו ערכים קריטיים ומילוי נתונים יהרוס את הניתוח נתונים
df_cleaned = df.dropna(subset=['price', 'city', 'state'])
dfLen =len(df)
print( f" {dfLen - len(df)} rows was deleted" , "\n")
# מילוי ערכים ריקים בעמודות  bed , bath , house_size , acre_lot בחציון
df['bed'] = df['bed'].fillna(df['bed'].median())
df['bath'] = df['bath'].fillna(df['bath'].median())
df['house_size'] = df['house_size'].fillna(df['house_size'].median())
df['acre_lot'] = df['acre_lot'].fillna(df['acre_lot'].median())
#מילוי בערך ברירת מחדל
df['brokered_by'] = df['brokered_by'].fillna('Unknown')
#המירו סוגי נתונים במידת הצורך
df['bed'] = df['bed'].astype('int64')
df['zip_code'] = df['zip_code'].astype(str)
#מילוי בערך ברירת מחדל לZIP
def fill_zip_by_city(group):
    mode = group.mode()
    if not mode.empty:
        return group.fillna(mode.iloc[0])
    return group.fillna('Unknown')
df['zip_code'] = df.groupby('city')['zip_code'].transform(fill_zip_by_city)
#שנו את שמות העמודות במידת הצורך
df = df.rename(columns={'bed': 'bedrooms', 'bath': 'bathrooms', 'acre_lot': 'lot_size_acres'})
# המרת התאריכים לצורת כתיבת תאריך מוכרת לפייתון
df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')
print("info printing")
df.info()

#שלב 4
# סינון רק של נכסים שהסטטוס שלהם הוא מכור ('sold')
df_sold = df[df['status'] == 'sold']

# יצירת טבלת ציר לפי: מדינה, עיר, וסוכן/משרד תיווך
pivot_agent_sales = pd.pivot_table(
    df_sold,
    index=['state', 'city', 'brokered_by'],
    values=['price'],
    aggfunc={'price': ['count', 'mean', 'max']}, # כמות מכירות, מחיר ממוצע, ומחיר מקסימלי למכירה
    fill_value=0
)
print(pivot_agent_sales.head(80))
simple_pivot = pd.pivot_table(
    df,
    index=['state', 'status'],          # שורות: מדינה וסטטוס הנכס
    values=['price', 'house_size'],     # ערכים: מחיר ושטח הבית
    aggfunc='mean'                      # חישוב ממוצע בלבד
).reset_index()

# הצגת 15 השורות הראשונות
print(simple_pivot.head(15))

city_summary_pivot = pd.pivot_table(
  df,
    index=['state'],                   # שורות: העיר
    values=['price'],                 # ערכים: מחיר
    aggfunc=['mean', 'max'])         # ממוצע ומקסימום
print(city_summary_pivot.head(15))

#שלב 5
#1
#Average Price vs Maximum Price per Agent
df_agents = pivot_agent_sales.reset_index()
df_agents.columns = ['state', 'city', 'brokered_by', 'sales_count', 'max_price', 'avg_price']
df_agents.plot(kind='scatter', x='avg_price', y='max_price', color='purple', alpha=0.6, figsize=(8, 5))
plt.title('Average Price vs Maximum Price per Agent')
plt.xlabel('Average Price')
plt.ylabel('Max Price')
plt.tight_layout()
plt.savefig("Average Price vs Maximum Price per Agent .png")
plt.show()
#2
#Top Agents Market Share by Sales
df_agents = pivot_agent_sales.reset_index().sort_values(by=pivot_agent_sales.columns[0], ascending=False).head(8)
agent_col = 'brokered_by' if 'brokered_by' in df_agents.columns else df_agents.columns[1]
sales_col = df_agents.select_dtypes(include='number').columns[0]
plt.figure(figsize=(9, 9))
plt.pie(
    df_agents[sales_col],
    labels=df_agents[agent_col],
    startangle=90,
autopct='%1.1f%%',
)
plt.title('Top Agents Market Share by Sales ')
plt.tight_layout()
plt.savefig("Top Agents Market Share by Sales .png")
plt.show()
#3
#Average Price by State and Status
df_simple_pivot = simple_pivot.reset_index()
numeric_cols = df_simple_pivot.select_dtypes(include='number').columns
val_col = numeric_cols[0]
pivot_for_plot = df_simple_pivot.pivot(index='state', columns='status', values=val_col)
pivot_for_plot = pivot_for_plot.apply(pd.to_numeric, errors='coerce')
pivot_for_plot.plot(kind='bar', figsize=(10, 5), color=['skyblue', 'orange'  , 'pink'])
plt.title('Average Price by State and Status')
plt.xlabel('State')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Average Price by State and Status.png")
plt.show()
#4
#Average House Size by State
df_simple_pivot = simple_pivot.reset_index()
numeric_cols = df_simple_pivot.select_dtypes(include='number').columns
size_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
df_simple_pivot[size_col] = pd.to_numeric(df_simple_pivot[size_col], errors='coerce')
df_simple_pivot.plot(
    kind='line',
    x='state',
    y=size_col,
    marker='o', color='forestgreen', linewidth=2, figsize=(10, 5))
plt.title('Average House Size by State')
plt.xlabel('State')
plt.ylabel('Average House Size')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Average House Size by State.png")
plt.show()
#5
#Top 5 Cities Max Sales
df = city_summary_pivot.reset_index()
df_top5 = df.sort_values(by=df.columns[1], ascending=False).head(5)
plt.figure(figsize=(8, 4))
plt.barh(df_top5.iloc[:, 0], df_top5.iloc[:, 2], color='orange')
plt.title('Top 5 Cities Max Sales')
plt.xlabel('Max Sale')
plt.ylabel('City')
plt.gca().invert_yaxis()
plt.savefig("Top 5 Cities Max Sales.png")
plt.show()
#6
#Top 5 Cities
df = city_summary_pivot.reset_index()
df_top5 = df.sort_values(by=df.columns[1], ascending=False).head(5)
plt.figure(figsize=(7, 4))
plt.bar(df_top5.iloc[:, 0], df_top5.iloc[:, 1], color='purple')
plt.title('Top 5 Cities')
plt.xlabel('City')
plt.ylabel('Sales')
plt.savefig("Top 5 Cities.png")
plt.show()
#7
# Boxplot - Data Distribution
plt.figure(figsize=(7, 4))
plt.boxplot(df.select_dtypes(include='number').iloc[:, 0].dropna(), patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title('Boxplot - Data Distribution')
plt.ylabel('Values')
plt.savefig("Boxplot - Data Distribution.png")
plt.show()
#8
#Density Plot - Smooth Data Distribution
data_to_plot = df.select_dtypes(include='number').iloc[:, 0].dropna()
plt.figure(figsize=(8, 4))
sns.kdeplot(data_to_plot, fill=True, color='teal', alpha=0.4)
plt.title('Density Plot - Smooth Data Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("Density Plot - Smooth Data Distribution.png")
plt.show()
#9
#Violin Plot - Data Distribution and Density
plt.figure(figsize=(7, 4))
sns.violinplot(y=data_to_plot, color='plum', inner='box')
plt.title('Violin Plot - Data Distribution and Density')
plt.ylabel('Value')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("Violin Plot - Data Distribution and Density.png")
plt.show()
#10
# גרף plotly - 15 המדינות היקרות ביותר לפי מחיר ממוצע (plotly היא הספרייה השלישית הנדרשת בשלב 5)
# מסננים ערכים חריגים מקולקלים לפני החישוב כדי לקבל גרף קריא
df_valid = df[df['price'].between(1e4, 1e7)]
top_states = df_valid.groupby('state')['price'].mean().sort_values(ascending=False).head(15).reset_index()
fig_plotly = px.bar(
    top_states,
    x='state',
    y='price',
    color='price',
    title='Top 15 States by Average Price (Plotly)',
    labels={'price': 'Average Price ($)', 'state': 'State'},
    color_continuous_scale='Viridis'
)
fig_plotly.update_layout(xaxis_tickangle=-45)
# שמירה כ-HTML תמיד עובדת; אפשר לפתוח בדפדפן ולצלם צילום מסך להכנסה לוורד
fig_plotly.write_html("Top 15 States by Average Price (Plotly).html")
# שמירה כתמונת PNG דורשת את החבילה kaleido (pip install kaleido). אם לא מותקנת - מדלגים בלי לקרוס
try:
    fig_plotly.write_image("Top 15 States by Average Price (Plotly).png")
except Exception as e:
    print("להצגת הגרף כ-PNG יש להתקין kaleido (pip install kaleido). נשמר בינתיים כ-HTML.")
fig_plotly.show()

#שלב 6
# קיבוץ לאשכולות לפי המחיר הממוצע והמקסימלי של כל מדינה (מתוך טבלת הציר city_summary_pivot)
X = city_summary_pivot.copy()
X.columns = [str(col) for col in X.columns]   # שמות העמודות: ('mean', 'price') ו-('max', 'price')
X = X.dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
inertia = []
for k in range(1, 11):
  model = KMeans(n_clusters=k, random_state=42, n_init=10)
  model.fit(X_scaled)
  inertia.append(model.inertia_)
plt.figure(figsize=(6, 3))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.savefig("Elbow Method.png")
plt.show()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
X['Cluster'] = kmeans.fit_predict(X_scaled)
plt.figure(figsize=(6, 4))
plt.scatter(
    X.iloc[:, 0], X.iloc[:, 1], c=X['Cluster'], cmap='viridis', edgecolor='k'
)
plt.title('K-Means Clustering')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.savefig("K-Means Clustering.png")
plt.show()
print(X.groupby('Cluster').mean())

# שלב 7
# רגרסיה לינארית בין המחיר הממוצע (X) למחיר המקסימלי (Y) של כל מדינה (מתוך city_summary_pivot)
df_reg = city_summary_pivot.copy()
df_reg.columns = [str(col) for col in df_reg.columns]
df_reg = df_reg.dropna()
X = df_reg.iloc[:, 0]   # ('mean', 'price') - מחיר ממוצע
Y = df_reg.iloc[:, 1]   # ('max', 'price')  - מחיר מקסימלי
X_const = sm.add_constant(X)
model = sm.OLS(Y, X_const).fit()
print(model.summary())
plt.figure(figsize=(6, 4))
plt.scatter(X, Y, color='blue', alpha=0.5, label='נתונים')
plt.plot(X, model.predict(X_const), color='red', linewidth=2, label='Regression Line')
plt.title('Linear Regression')
plt.xlabel('X (Independent Variable)')
plt.ylabel('Y (Dependent Variable)')
plt.legend()
plt.savefig("Linear Regression.png")
plt.show()

# שלב 9
# הכנת הנתונים ומשתנה היעד
df_cls = pd.read_csv("DB/realtor-data1.csv", encoding='latin1').dropna(subset=['price'])
df_cls = df_cls.rename(columns={'bed': 'bedrooms', 'bath': 'bathrooms', 'acre_lot': 'lot_size_acres'})
df_cls['target'] = (df_cls['price'] > df_cls['price'].median()).astype(int)

X = df_cls[['bedrooms', 'bathrooms', 'lot_size_acres', 'house_size']].dropna()
y = df_cls.loc[X.index, 'target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# עצי החלטה - Gini ו-Entropy
dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42).fit(X_train, y_train)
dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42).fit(X_train, y_train)

# הצגת העצים
plt.figure(figsize=(8, 4))
plot_tree(dt_gini, feature_names=X.columns, filled=True)
plt.title('Decision Tree - Gini')
plt.show()

plt.figure(figsize=(8, 4))
plot_tree(dt_entropy, feature_names=X.columns, filled=True)
plt.title('Decision Tree - Entropy')
plt.show()

# הערכת מודל עץ ההחלטה
y_pred_dt = dt_gini.predict(X_test)
y_prob_dt = dt_gini.predict_proba(X_test)[:, 1]

print("--- Decision Tree (Gini) ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt), "\n")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Recall:", recall_score(y_test, y_pred_dt))
print("AUC:", roc_auc_score(y_test, y_prob_dt))
print("Sample Predictions:", dt_gini.predict(X_test[:3]), "\n")

# בונוס: העברת מספר נכסים לדוגמה מהקובץ דרך עץ ההחלטה ובדיקת התוצאה מול הערך האמיתי
print("--- Bonus: Sample rows through the Decision Tree ---")
sample = X_test.head(5).copy()
sample['predicted_target'] = dt_gini.predict(X_test.head(5))
sample['actual_target'] = y_test.head(5).values
print(sample)
print("target = 1 => מחיר מעל החציון, target = 0 => מחיר מתחת לחציון", "\n")

# Random Forest ו-XGBoost
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42).fit(X_train, y_train)
xgb = XGBClassifier(n_estimators=100, max_depth=3, random_state=42).fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

print("--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("AUC:", roc_auc_score(y_test, y_prob_rf), "\n")

print("--- XGBoost ---")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("AUC:", roc_auc_score(y_test, y_prob_xgb), "\n")

# רשת עצבית (TensorFlow)
nn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])

nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

loss, accuracy = nn.evaluate(X_test, y_test, verbose=0)
print("--- Neural Network ---")
print("Neural Network Accuracy:", accuracy)
