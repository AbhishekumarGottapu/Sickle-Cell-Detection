import joblib
from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import numpy as np
from keras.models import load_model
import cv2
import tensorflow as tf
from datetime import date

# Get the current date
current_date = date.today()
current_date=current_date.strftime("%d-%m-%Y")
# Print the current date


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))

with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('Remember me')


class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])


@app.route('/')
def index():
    return render_template("dashboard.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/help')
def help():
    return render_template("help.html")

@app.route('/upload')
@login_required
def upload():
    return render_template("upload.html")

@app.route('/scd_result')
def terms():
    return render_template("scd_result.html")

@app.route('/sickle')
@login_required
def scd():
    return render_template("sickle.html")

name = None


@app.route('/login', methods=['GET', 'POST'])
def login():
    global name
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                name = user.username  # Save the username in a variable called name
            # You can now redirect or use the 'name' variable as needed
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))
        return render_template("login.html", form=form, error="Invalid username or password.")
    return render_template("login.html", form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html', form=form)


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


model = load_model('sicklecelldetection.h5')


# # Define a function to make predictions
def preprocess_image(image):
    # Assuming the image is already a decoded numpy array
    image = tf.image.resize(image, [256, 256])
    image = image / 255.0  # Normalize the image to [0, 1]
    return image

def predict_scd(image):
    # Convert the image to a format suitable for input
    processed_image = preprocess_image(image)
    # Make predictions using the pre-trained model
    prediction = model.predict(np.expand_dims(processed_image, axis=0))[0]
    return prediction

@login_required
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        file = request.files['image']  # Access the uploaded file using 'image' key
        if file:
            try:
                image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                prediction = predict_scd(image)
                
                # Determine the message based on the prediction result
                if prediction > 0.5:
                    msg = "You don't have sickle cell"
                else:
                    msg = "You have sickle cell"
                
                # Render the template with the message
                return render_template('result.html', msg=msg)
            except Exception as e:
                return render_template('upload.html', error="Error processing image: {}".format(str(e)))
        else:
            return render_template('upload.html', error="No file selected.")
    return render_template('upload.html')


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==11):
        loaded_model = joblib.load('predict.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route("/predict",  methods=['GET', 'POST'])
@login_required
def predictscd():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        print(len(to_predict_list))
        if len(to_predict_list) == 11:
            result = ValuePredictor(to_predict_list, 11)
    if(result > 0.75):
        prediction = "Patient has a high risk of Sickle Cell Disease, please consult your doctor immediately"
        diet_recommendation = """
    Hydration: Drink at least 8-10 glasses of water per day to prevent dehydration, which can trigger SCD crises.
    Nutrient-Rich Foods: Focus on foods rich in vitamins C, E, and B-complex, as well as minerals like iron and magnesium. Examples include citrus fruits, dark leafy greens, nuts, seeds, lean meats, and whole grains.
    Supplementation: Consider supplements such as folic acid, vitamin D, and omega-3 fatty acids to support red blood cell production, bone health, and inflammation reduction, respectively.
    Avoidance of Triggers: Stay indoors during extreme temperatures, avoid high altitudes, and practice relaxation techniques to minimize stress-related complications.
    Regular Monitoring: Schedule regular check-ups with healthcare providers to monitor blood counts, oxygen levels, and organ function to detect any complications early.
    """
        medication_recommendation = """
    Medication Recommendations:
    - Hydroxyurea (Hydrea): Start with 15 mg/kg/day and increase gradually to a maximum of 35 mg/kg/day based on response and tolerance.
    - Opioid Pain Relievers: Morphine sulfate extended-release tablets (e.g., MS Contin) 30 mg every 12 hours as needed for severe pain during crises.
    - Antibiotics (Prophylactic): Penicillin V potassium 500 mg twice daily for adults or amoxicillin 250 mg twice daily for children to prevent infections.
    """
    elif result>0.25 or result<0.75:
        prediction = "Patient has a moderate risk of Sickle Cell Disease"
        diet_recommendation = """
    Balanced Diet: Include a variety of nutrient-dense foods such as fruits, vegetables, whole grains, lean proteins, and healthy fats to support overall health and reduce inflammation.
    Limiting Saturated Fat: Opt for lean protein sources like poultry, fish, legumes, and tofu instead of processed meats and fried foods to lower the risk of heart disease and stroke.
    Frequent Meals: Eat small, frequent meals throughout the day to maintain energy levels and prevent blood sugar fluctuations, which can trigger SCD symptoms.
    Iron-Rich Foods: Incorporate iron-rich foods like spinach, lentils, fortified cereals, and poultry to prevent anemia and fatigue.
    Avoidance of Alcohol and Smoking: Refrain from smoking and limit alcohol consumption to reduce the risk of complications such as vaso-occlusive crises and lung damage.
    """
        medication_recommendation = """
    Medication Recommendations:
    - Hydroxyurea (Hydrea): Similar to high severity, starting with 15 mg/kg/day and increasing as needed.
    - NSAIDs (Nonsteroidal Anti-Inflammatory Drugs): Ibuprofen 400 mg every 6 hours as needed for mild to moderate pain relief.
    - Folic Acid (Supplementation): 1 mg/day for adults and adjusted doses for children based on age and weight.
    """
        
    else:
        prediction="Patient has a low risk of Sickle Cell Disease"
        diet_recommendation = """
    Regular Exercise: Engage in regular, moderate-intensity exercise such as brisk walking, cycling, or swimming to improve circulation, strengthen muscles, and boost overall well-being.
    Moderate Salt Intake: Limit processed foods and restaurant meals high in sodium to maintain healthy blood pressure levels and reduce the risk of dehydration and kidney damage.
    Vitamin Supplements: Consider taking a daily multivitamin or specific supplements under the guidance of a healthcare provider to address any nutritional deficiencies and support immune function.
    Portion Control: Practice mindful eating and portion control to prevent overeating and maintain a healthy weight, which can reduce the risk of obesity-related complications.
    Stress Management: Incorporate stress-reduction techniques such as yoga, meditation, deep breathing exercises, or hobbies to promote relaxation and improve mental well-being.
    """
        medication_recommendation = """
    Medication Recommendations:
    - Folic Acid (Supplementation): Same as moderate severity, 1 mg/day for adults and adjusted doses for children.
    - NSAIDs (Nonsteroidal Anti-Inflammatory Drugs): Naproxen 220 mg twice daily as needed for mild pain relief.
    - Prophylactic Antibiotics: Penicillin VK 500 mg once daily for adults or amoxicillin 250 mg twice daily for children to prevent infections.
    """
    return render_template("scd_result.html", prediction_text=prediction,diet_recommendation=diet_recommendation,medication_recommendation=medication_recommendation,name=name,date=current_date)
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
