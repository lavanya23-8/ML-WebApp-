from flask import *
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
# Helps Flask locate static files and templates relative to the module
app = Flask(__name__)
# Sets the secret key used by Flask to sign session cookies and flash messages
app.secret_key = 'ni016'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/ml_app'
# Saves memory
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initializes Bcrypt so you can hash passwords
bcrypt = Bcrypt(app)

# Define a database model - User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    # Check if the request method is POST (form submission)
    if request.method == 'POST':
        # Read username, email, and password
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        # Hash the password for secure storage
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        # Create a new User object with form data
        new_user = User(username=username, email=email, password=hashed_password)

        try:
            # Insert new user into DB
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except:
            flash('Error: Username or email already exists.', 'danger')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Check if the request method is POST (form submission)
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        # Query the user by email
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            # Successful login
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password. Please try again.', 'danger')
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to access the dashboard.', 'warning')
        return redirect(url_for('login'))
    username = session['username']
    return render_template('dashboard.html', username=username)

@app.route('/logout', methods=['POST'])
def logout():
    # Remove user_id and username from session
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

UPLOAD_FOLDER = os.path.join(os.path.abspath(os.getcwd()), 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global uploaded_data
    file_uploaded = False
    columns, data = [], []
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'warning')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_uploaded = True
            flash('File uploaded successfully!', 'success')
            try:
                uploaded_data = pd.read_csv(file_path)
                columns = uploaded_data.columns.tolist()
                data = uploaded_data.head(10).values.tolist()
                return render_template('upload.html', file_uploaded=file_uploaded, columns=columns, data=data)
            except Exception as e:
                flash(f'Error reading the file: {e}', 'danger')
                return redirect(request.url)
            flash('Invalid file type. Please upload a CSV file.', 'danger')
            return redirect(request.url)
    return render_template('upload.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        inputs = {
            "Pregnancies": int(request.form['Pregnancies']),
            "Glucose": float(request.form['Glucose']),  # fixed naming
            "BloodPressure": float(request.form['BloodPressure']),
            "SkinThickness": float(request.form['SkinThickness']),
            "Insulin": float(request.form['Insulin']),
            "BMI": float(request.form['BMI']),
            "DiabetesPedigreeFunction": float(request.form['DiabetesPedigreeFunction']),
            "Age": float(request.form['Age']),
        }
        input_values = list(inputs.values())
        model = joblib.load("models/model.pkl")
        prediction = model.predict([input_values])[0]
        result = "Diabetes Detected" if prediction == 1 else "No Diabetes Detected"
        return render_template("result.html", inputs=inputs, result=result)
    return render_template("predict.html")
@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    global uploaded_data
    if uploaded_data is None:
        flash('No dataset uploaded.', 'warning')
        return redirect(url_for('upload'))
    
    scatter_plot = generate_scatter_plot(uploaded_data)
    return render_template("visualize.html", scatter_plot=scatter_plot)

def generate_scatter_plot(data):
    data['Pregnancies'] = data['Pregnancies'].apply(lambda x: int(x) if not pd.isnull(x) else 0)
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    if len(numerical_columns) >= 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data['Pregnancies'], y=data['Glucose'])
        plt.title("Scatter plot: Pregnancies vs Glucose")
        plt.xlabel("Pregnancies")
        plt.ylabel("Glucose")
        max_pregnancies = data['Pregnancies'].max()
        plt.xticks(ticks=np.arange(0, max_pregnancies + 1, 1))
        filepath = os.path.join('static', 'images', 'scatter_plot.png')
        plt.savefig(filepath)
        plt.close()
        return filepath
    return None
        
@app.route('/train', methods=['GET', 'POST'])
def train():
    global uploaded_data
    if uploaded_data is None:
        flash('No data uploaded.', 'warning')
        return redirect(url_for('upload'))
    try:
        X = uploaded_data.iloc[:, :-1]
        y = uploaded_data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump(rf_model, "models/model.pkl")
        #Evalute model
        y_pred=rf_model.predict(X_test)
        acc=accuracy_score(y_test,y_pred)
        acc_percent = round(acc * 100)
        #Plot confusion matrix
        cm=confusion_matrix(y_test,y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.title(f"Confusion Matrix(Accuracy:{acc_percent}%)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_path = os.path.join('static', 'images', 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        flash(f'Training successful! Accuracy:{acc_percent}%', 'success')
        return render_template('train.html',accuracy=acc_percent,confusion_matrix=cm_path)
    except Exception as e:
        flash(f'Error during training: {e}', 'danger')
        return redirect(url_for('upload'))


if __name__ == '__main__':
    # Ensure tables are created
    with app.app_context():
        db.create_all()
    # Run the app
    app.run(debug=False, host='0.0.0.0', port=5000)
