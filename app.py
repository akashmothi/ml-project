import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from config import DB_CONFIG

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
VISUALIZATION_FOLDER = "static/visualizations"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["VISUALIZATION_FOLDER"] = VISUALIZATION_FOLDER
ALLOWED_EXTENSIONS = {'csv'}

MODEL_PATHS = {
    "knn": "models/knn_model.pkl",
    "linear_regression": "models/lr_model.pkl",
    "kmeans": "models/kmeans_model.pkl",
    "naive_bayes": "models/naive_bayes_model.pkl",
    "svm": "models/svm_model.pkl",
    "cnn": "models/cnn_model.h5",
    "rnn": "models/rnn_model.h5"
}

# ✅ Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ✅ Database Connection

def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.errors.ProgrammingError as e:
        if e.errno == 1049:  # Error 1049: Database does not exist
            conn = mysql.connector.connect(
                host=DB_CONFIG['host'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password']
            )
            cursor = conn.cursor()
            cursor.execute("CREATE DATABASE ml_database")  # Create the database
            cursor.close()
            conn.close()
            return mysql.connector.connect(**DB_CONFIG)  # Reconnect after creating DB
        else:
            raise 


# ✅ Store Uploaded Data in MySQL
def store_data_in_mysql(df):
    conn = get_db_connection()
    cursor = conn.cursor()

    # ✅ Ensure table exists
    create_table_query = """
    CREATE TABLE IF NOT EXISTS uploaded_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        feature1 FLOAT, 
        feature2 FLOAT,
        feature3 FLOAT,
        feature4 FLOAT
    )
    """
    cursor.execute(create_table_query)

    # ✅ Insert data into MySQL
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO uploaded_data (feature1, feature2, feature3, feature4)
            VALUES (%s, %s, %s, %s)
        """, tuple(row[:4]))

    conn.commit()
    cursor.close()
    conn.close()
    
def fetch_old_dataset():
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM uploaded_data"
    cursor.execute(query)

    columns = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=columns)

    cursor.close()
    conn.close()
    
    return df


# ✅ Flask Route for Uploading and Predicting
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_algorithm = request.form.get("algorithm")
        file = request.files['file']

        if not file or file.filename == '':
            return render_template('index.html', error="No file selected!")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            df = pd.read_csv(filepath)
            store_data_in_mysql(df)

            if selected_algorithm == "apriori":
                prediction_results, plot_path, accuracy = run_apriori(df)
            elif selected_algorithm in MODEL_PATHS:
                prediction_results, plot_path, accuracy = predict_model(selected_algorithm, df)
            else:
                return render_template('index.html', error="Selected model not found!")

            return render_template('results.html', prediction_results=prediction_results, plot_path=plot_path, accuracy=accuracy)

    return render_template('index.html')

# ✅ ML Model Predictions with Accuracy Calculation
def predict_model(algorithm, df):
    if df.shape[1] < 4:
        return "CSV must have at least 4 feature columns.", None, None

    features_array = df.iloc[:, :4]

    if algorithm in ["cnn", "rnn"]:
        # ✅ Load CNN or RNN Model
        if not os.path.exists(MODEL_PATHS[algorithm]):
            return f"Error: {algorithm.upper()} model file is missing!", None, None

        model = tf.keras.models.load_model(MODEL_PATHS[algorithm])
        features_array = features_array.to_numpy().reshape(features_array.shape[0], 4, 1)  # Reshape for CNN & RNN
        predictions = np.argmax(model.predict(features_array), axis=1)

        df['Prediction'] = predictions
        plot_path = visualize_results(df, algorithm)
        return df.to_html(classes="styled-table"), plot_path, "N/A"  # CNN/RNN do not have accuracy

    # ✅ Handle Other ML Models
    if not os.path.exists(MODEL_PATHS[algorithm]):
        return f"Error: {algorithm.upper()} model file is missing!", None, None

    with open(MODEL_PATHS[algorithm], "rb") as model_file:
        model = pickle.load(model_file)

    # Splitting data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(features_array, df.iloc[:, -1], test_size=0.2, random_state=42)

    model.fit(X_train, y_train)  # Train model
    predictions = model.predict(X_test)  # Predict on test data

    # ✅ Accuracy Calculation
    if algorithm == "linear_regression":
        accuracy = f"Mean Squared Error: {mean_squared_error(y_test, predictions):.4f}"
    else:
        accuracy = f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%"

    df['Prediction'] = model.predict(features_array)
    plot_path = visualize_results(df, algorithm)

    return df.to_html(classes="styled-table"), plot_path, accuracy

# ✅ Apriori Algorithm with Visualization
# ✅ Enhanced Apriori Algorithm with Custom Rule Validation
def run_apriori(df):
    df = df.astype(bool)  # Convert to boolean for one-hot encoding
    frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    if rules.empty:
        return "No strong association rules found.", None, "N/A"

    # ✅ Filter Rules: `1*2` (1 antecedent, 2 consequents)
    one_two_rules = rules[(rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 2)]

    # ✅ Filter Rules: `2*#` (2 antecedents, any consequents)
    two_any_rules = rules[rules["antecedents"].apply(len) == 2]

    # ✅ Merge and sort rules
    filtered_rules = pd.concat([one_two_rules, two_any_rules]).sort_values(by="lift", ascending=False)

    if filtered_rules.empty:
        return "No rules match the required constraints.", None, "N/A"

    plot_path = visualize_results(filtered_rules, "apriori")
    return filtered_rules.to_html(classes="styled-table"), plot_path, "N/A"

# ✅ Data Visualization for All Models
def visualize_results(df, algorithm):
    plt.figure(figsize=(8, 5))

    if algorithm == "kmeans":
        sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df['Prediction'], palette="viridis")
        plt.title("K-Means Clustering Results")

    elif algorithm == "knn":
        sns.countplot(x=df["Prediction"], palette="coolwarm")
        plt.title("K-NN - Class Distribution")

    elif algorithm == "svm":
        # ✅ Updated Visualization for SVM - Decision Boundary
        from mlxtend.plotting import plot_decision_regions
        X = df.iloc[:, :2].values  # Take first 2 features for visualization
        y = df["Prediction"].astype(int).values  # Convert labels to int

        # Create a dummy model for visualization
        from sklearn.svm import SVC
        model = SVC(kernel="linear")  # Linear kernel for visualization
        model.fit(X, y)

        plt.figure(figsize=(8, 5))
        plot_decision_regions(X, y, clf=model, legend=2)
        plt.title("SVM Decision Boundary")

    elif algorithm == "naive_bayes":
        sns.countplot(x=df["Prediction"], palette="coolwarm")
        plt.title("Naive Bayes - Class Distribution")

    elif algorithm == "linear_regression":
        plt.scatter(df.iloc[:, 0], df["Prediction"], color="blue")
        plt.xlabel("Feature 1")
        plt.ylabel("Predicted Value")
        plt.title("Linear Regression Predictions")

    elif algorithm == "apriori":
        top_rules = df.sort_values(by="lift", ascending=False).head(10)
        top_rules["rule"] = top_rules["antecedents"].astype(str) + " → " + top_rules["consequents"].astype(str)
        plt.barh(top_rules["rule"], top_rules["lift"], color="skyblue")
        plt.xlabel("Lift")
        plt.ylabel("Association Rule")
        plt.title("Top 10 Apriori Rules")

    plot_filename = f"{algorithm}_plot.png"
    plot_path = os.path.join(app.config["VISUALIZATION_FOLDER"], plot_filename)
    plt.savefig(plot_path)
    plt.close()

    return plot_filename

if __name__ == '__main__':
    app.run(debug=False)
