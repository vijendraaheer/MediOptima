from flask import Flask, render_template, request, redirect, url_for, session, send_file
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import pandas as pd
import matplotlib
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import io
import os
import sys
from dotenv import load_dotenv

pio.templates.default = "plotly_white"
load_dotenv()

# ================= AI MODULE =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..', 'ml_models', 'ai_insights'))
from AI_Insights import generate_insight

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_key")

# ================= DATABASE =================
db = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST", "localhost"),
    user=os.getenv("MYSQL_USER", "root"),
    password=os.getenv("MYSQL_PASSWORD", ""),
    database=os.getenv("MYSQL_DB", "medioptima"),
    port=int(os.getenv("MYSQL_PORT", 3306))
)

cursor = db.cursor(dictionary=True)

# ================= LOAD DATA =================
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def path(file):
        return os.path.join(BASE_DIR, '..', 'data', file)

    old = pd.read_csv(path('processed/clean_hospital_data.csv'))
    pred = pd.read_csv(path('outputs/Prediction.csv'))
    bed = pd.read_csv(path('outputs/Bed_Requirement.csv'))
    staff = pd.read_csv(path('outputs/Optimized_Staff.csv'))
    risk = pd.read_csv(path('outputs/Surge_Outbreak_Alerts.csv'))

    for dataset in [old, pred, bed, staff, risk]:
        dataset['Date'] = pd.to_datetime(dataset['Date'])

    df = pred.merge(bed, on='Date', how='left') \
             .merge(staff, on='Date', how='left') \
             .merge(risk, on='Date', how='left')

    df = df.loc[:, ~df.columns.duplicated()]

    return old, df

old, df = load_data()

# ================= ROLE DECORATOR =================
def role_required(roles):
    def wrapper(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if 'user' not in session:
                return redirect(url_for('login'))
            if session['user']['role'] not in roles:
                return "Access Denied"
            return f(*args, **kwargs)
        return decorated
    return wrapper

# ================= HOME =================
@app.route('/')
def home():
    return redirect(url_for('login'))

# ================= REGISTER =================
@app.route('/register', methods=['GET','POST'])
def register():

    if request.method == 'POST':

        cursor.execute(
            "INSERT INTO users (name,email,password,role,approved) VALUES (%s,%s,%s,%s,0)",
            (
                request.form['name'],
                request.form['email'],
                generate_password_hash(request.form['password']),
                request.form['role']
            )
        )

        db.commit()

        return "Registered successfully. Wait for admin approval."

    return render_template("register.html")

# ================= LOGIN =================
@app.route('/login', methods=['GET','POST'])
def login():

    if request.method == 'POST':

        cursor.execute(
            "SELECT * FROM users WHERE email=%s",
            (request.form['email'],)
        )

        user = cursor.fetchone()

        if user and check_password_hash(user['password'], request.form['password']):

            if user['approved'] == 1:

                session['user'] = user

                return redirect(url_for('dashboard'))

            else:
                return "Account not approved."

        return "Invalid credentials."

    return render_template("login.html")

# ================= LOGOUT =================
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ================= ADMIN APPROVAL =================
@app.route('/approve/<int:user_id>')
@role_required(['admin'])
def approve(user_id):

    cursor.execute("UPDATE users SET approved=1 WHERE id=%s", (user_id,))
    db.commit()

    return redirect(url_for('dashboard'))

@app.route('/cancel/<int:user_id>')
@role_required(['admin'])
def cancel(user_id):

    cursor.execute("DELETE FROM users WHERE id=%s", (user_id,))
    db.commit()

    return redirect(url_for('dashboard'))

# ================= DASHBOARD =================
@app.route('/dashboard', methods=['GET','POST'])
@role_required(['admin','doctor','staff'])
def dashboard():

    role = session['user']['role']
    selected_date = request.form.get('date')

    if selected_date:
        selected_date = pd.to_datetime(selected_date)
        latest = df[df['Date'] == selected_date].iloc[0]
    else:
        latest = df.iloc[-1]
        selected_date = latest['Date']

    selected_date_str = selected_date.strftime("%Y-%m-%d")

    # ================= KPIs =================

    admin_kpis = {
        "Total Daily Cost": latest.get("Total_Minimum_Daily_Cost",0),
        "Doctor Cost": latest.get("Doctor_Cost",0),
        "General Nurse Cost": latest.get("General_Nurse_Cost",0),
        "ICU Nurse Cost": latest.get("ICU_Nurse_Cost",0),
        "Beds Over Capacity": latest.get("Beds_Over_Capacity",0),
        "Z-score": latest.get("Z_score",0),
        "Outbreak Alert": latest.get("Outbreak_Alert",""),
        "Surge Alert": latest.get("Surge_Alert","")
    }

    doctor_kpis = {
        "Predicted Patients": latest.get("Predicted_Patients_x",0),
        "Expected Discharge": latest.get("Expected_discharge",0),
        "Predicted Emergency Cases": latest.get("Predicted_Emergency_Cases",0),
        "Predicted ICU Admissions": latest.get("Predicted_ICU_Admissions",0),
        "Required Beds": latest.get("Required_Beds",0),
        "ICU Beds": latest.get("ICU_Beds",0),
        "General Ward Beds": latest.get("General_Ward_Beds",0),
        "Bed Shortage Alert": latest.get("Bed_Shortage_Alert",False),
        "Emergency Margin": latest.get("emergency_margin",0)
    }

    staff_kpis = {
        "Doctors Required": latest.get("Doctors_Required",0),
        "General Nurses Required": latest.get("General_Nurses_Required",0),
        "ICU Nurses Required": latest.get("ICU_Nurses_Required",0)
    }

    # ================= COST =================

    daily_cost = latest['Total_Minimum_Daily_Cost']
    weekly_cost = df.tail(7)['Total_Minimum_Daily_Cost'].sum()
    monthly_cost = df.tail(30)['Total_Minimum_Daily_Cost'].sum()

    # ================= FORECAST GRAPH =================

    forecast_df = df.tail(30).sort_values('Date')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Predicted_Patients_x'],
        mode='lines+markers',
        name='Predicted Patients'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Predicted_Emergency_Cases'],
        mode='lines',
        name='Emergency Cases'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Predicted_ICU_Admissions'],
        mode='lines',
        name='ICU Admissions'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Required_Beds'],
        mode='lines',
        name='Required Beds'
    ))

    fig.update_layout(
        title="📊 30-Day Hospital Forecast",
        height=600,
        hovermode="x unified"
    )

    graph_html = pio.to_html(fig, full_html=False)

    # ================= AI INSIGHTS =================

    report = generate_insight(old, df)

    bed_alert = ""
    icu_alert = ""

    for line in report.split("\n"):

        if "Bed Occupancy" in line:
            bed_alert = line

        if "ICU Usage" in line:
            icu_alert = line

    optimization = "Reassign staff or open additional beds if alerts are red."

    # ================= ADMIN USER MANAGEMENT =================

    pending = []
    approved = []

    if role == "admin":

        cursor.execute("SELECT * FROM users WHERE approved=0")
        pending = cursor.fetchall()

        cursor.execute("SELECT * FROM users WHERE approved=1")
        approved = cursor.fetchall()

    # ================= ROLE BASED DASHBOARD =================

    if role == "admin":

        return render_template(
            "admin_dashboard.html",
            selected_date=selected_date_str,
            admin_kpis=admin_kpis,
            daily_cost=daily_cost,
            weekly_cost=weekly_cost,
            monthly_cost=monthly_cost,
            graph=graph_html,
            bed_alert=bed_alert,
            icu_alert=icu_alert,
            report=report,
            optimization=optimization,
            pending=pending,
            approved=approved
        )

    elif role == "doctor":

        return render_template(
            "doctor_dashboard.html",
            selected_date=selected_date_str,
            doctor_kpis=doctor_kpis,
            graph=graph_html,
            report=report,
            bed_alert=bed_alert,
            icu_alert=icu_alert
        )

    else:

        return render_template(
            "staff_dashboard.html",
            selected_date=selected_date_str,
            staff_kpis=staff_kpis,
            graph=graph_html,
            report=report
        )

# ================= patients =================
@app.route('/patients')
@role_required(['admin','doctor'])
def patients():
    patient_data = df[['Date','Predicted_Patients_x','Expected_discharge','Predicted_Emergency_Cases','Predicted_ICU_Admissions']].tail(30)
    return render_template("patients.html", data=patient_data.to_dict(orient="records"))

# ================= BEDS =================
@app.route('/beds')
@role_required(['admin','doctor','staff'])
def beds():
    bed_data = df[['Date','Required_Beds','ICU_Beds','General_Ward_Beds','Bed_Shortage_Alert','Beds_Over_Capacity']].tail(30)
    return render_template("beds.html", data=bed_data.to_dict(orient="records"))

# ================= DOCTORS =================
@app.route('/doctors')
@role_required(['admin'])
def doctors():
    doctor_data = df[['Date','Doctors_Required','General_Nurses_Required','ICU_Nurses_Required']].tail(30)
    return render_template("doctors.html", data=doctor_data.to_dict(orient="records"))

# ================= APPOINTMENTS =================
@app.route('/appointments')
@role_required(['admin','doctor'])
def appointments():
    return render_template("appointments.html")

# ================= SETTINGS =================
@app.route('/settings')
@role_required(['admin'])
def settings():
    return render_template("settings.html")

# ================= APPROVE USERS PAGE =================
@app.route('/approve_page')
@role_required(['admin'])
def approve_page():

    cursor.execute("SELECT * FROM users WHERE approved=0")
    pending = cursor.fetchall()

    cursor.execute("SELECT * FROM users WHERE approved=1")
    approved = cursor.fetchall()

    return render_template(
        "approve_users.html",
        pending=pending,
        approved=approved
    )

# ================= FORECAST PAGE =================
# ================= FORECAST PAGE =================
@app.route('/forecast')
@role_required(['admin','doctor','staff'])
def forecast():

    forecast_df = df.tail(30).sort_values('Date')

    tomorrow_patients = int(forecast_df['Predicted_Patients_x'].iloc[-1])
    icu_pred = int(forecast_df['Predicted_ICU_Admissions'].iloc[-1])
    emergency_pred = int(forecast_df['Predicted_Emergency_Cases'].iloc[-1])

    alert = None
    if icu_pred > 20:
        alert = "⚠ ICU capacity may exceed safe limit."

    # ---------- FORECAST GRAPH ----------

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Predicted_Patients_x'],
        mode='lines+markers',
        name='Predicted Patients'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Predicted_Emergency_Cases'],
        mode='lines',
        name='Emergency Cases'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Predicted_ICU_Admissions'],
        mode='lines',
        name='ICU Admissions'
    ))

    fig.update_layout(
        title="Hospital Forecast (Next 30 Days)",
        height=500
    )

    forecast_graph = pio.to_html(fig, full_html=False)

    # ---------- HEATMAP ----------

    heatmap_fig = go.Figure(data=go.Heatmap(
        z=[forecast_df['Required_Beds']],
        x=forecast_df['Date'],
        y=["Bed Demand"],
        colorscale="RdYlGn_r"
    ))

    heatmap_fig.update_layout(
        title="AI Bed Demand Heatmap",
        height=300
    )

    heatmap_graph = pio.to_html(heatmap_fig, full_html=False)
    
    # ================= SURGE DETECTION =================

    avg_patients = int(df['Predicted_Patients_x'].mean())
    tomorrow_patients = int(forecast_df['Predicted_Patients_x'].iloc[-1])

    surge_alert = None
    recommendation = None

    if tomorrow_patients > avg_patients * 1.25:

        surge_alert = "⚠ AI Surge Alert: Patient demand expected to spike."

        extra_beds = int((tomorrow_patients - avg_patients) * 0.6)
        extra_doctors = int(extra_beds / 5)
        extra_nurses = int(extra_beds / 2)

        recommendation = f"""
        Recommended Actions:
        Open {extra_beds} extra beds,
        Assign {extra_doctors} additional doctors,
        Deploy {extra_nurses} extra nurses
        """

    return render_template(
        "forecast.html",
        forecast_graph=forecast_graph,
        heatmap_graph=heatmap_graph,
        tomorrow_patients=tomorrow_patients,
        icu_pred=icu_pred,
        emergency_pred=emergency_pred,
        alert=alert,
        surge_alert=surge_alert,
        recommendation=recommendation,
        data=forecast_df.to_dict(orient="records")
    )
    
# ================= DOWNLOAD PDF =================

@app.route('/download_report')
@role_required(['admin','doctor','staff'])
def download_report():

    buffer = io.BytesIO()

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4
    )

    elements = []

    # -------------------------
    # HEADER
    # -------------------------

    elements.append(Paragraph("MediOptima Hospital", styles['Title']))
    elements.append(Paragraph("AI Powered Resource Optimization Report", styles['Heading2']))
    elements.append(Spacer(1,20))

    date_now = datetime.now().strftime("%d %B %Y")

    elements.append(Paragraph(f"<b>Report Generated:</b> {date_now}", styles['Normal']))
    elements.append(Spacer(1,20))


    # -------------------------
    # KPI SUMMARY
    # -------------------------

    latest = df.iloc[-1]

    patients = int(latest['Predicted_Patients_x'])
    beds = int(latest['Required_Beds'])
    doctors = int(latest['Doctors_Required'])
    nurses = int(latest['General_Nurses_Required']) + int(latest['ICU_Nurses_Required'])

    occupancy = round((patients/beds)*100,2) if beds > 0 else 0

    kpi_data = [
        ["Metric","Value"],
        ["Predicted Patients", patients],
        ["Required Beds", beds],
        ["Doctors Required", doctors],
        ["Total Nurses Required", nurses],
        ["Bed Occupancy Forecast", f"{occupancy}%"]
    ]

    table = Table(kpi_data)

    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.darkblue),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),1,colors.grey),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("ALIGN",(1,1),(-1,-1),"CENTER")
    ]))

    elements.append(Paragraph("Hospital KPI Summary", styles['Heading3']))
    elements.append(Spacer(1,10))
    elements.append(table)

    elements.append(Spacer(1,30))


    # -------------------------
    # FORECAST INSIGHT
    # -------------------------

    insight = generate_insight(old, df)

    elements.append(Paragraph("AI Forecast Insight", styles['Heading3']))
    elements.append(Spacer(1,10))
    elements.append(Paragraph(insight.replace("\n","<br/>"), styles['Normal']))

    elements.append(Spacer(1,30))


    # -------------------------
    # FORECAST TABLE
    # -------------------------

    forecast_data = df[['Date','Predicted_Patients_x','Required_Beds']].tail(7)

    table_data = [["Date","Predicted Patients","Required Beds"]]

    for _,row in forecast_data.iterrows():

        table_data.append([
            row['Date'].strftime("%d-%m-%Y"),
            int(row['Predicted_Patients_x']),
            int(row['Required_Beds'])
        ])

    forecast_table = Table(table_data)

    forecast_table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.grey),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),1,colors.black)
    ]))

    elements.append(Paragraph("7 Day Forecast Overview", styles['Heading3']))
    elements.append(Spacer(1,10))
    elements.append(forecast_table)

    elements.append(Spacer(1,30))


    # -------------------------
    # SYSTEM SUMMARY
    # -------------------------

    summary = """
    MediOptima is an AI powered hospital resource optimization platform
    designed to forecast patient demand and optimize allocation of
    hospital resources including beds, doctors and nursing staff.

    The system analyzes historical healthcare data and generates
    predictive insights that help hospital administrators prevent
    overcrowding, staff shortages and resource wastage.
    """

    elements.append(Paragraph("System Summary", styles['Heading3']))
    elements.append(Spacer(1,10))
    elements.append(Paragraph(summary, styles['Normal']))


    # -------------------------
    # FOOTER
    # -------------------------

    elements.append(Spacer(1,40))
    elements.append(Paragraph("Generated by MediOptima AI System", styles['Italic']))


    # BUILD PDF

    doc.build(elements)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="MediOptima_AI_Report.pdf",
        mimetype='application/pdf'
    )
    
# ================= RUN =================

if __name__ == '__main__':
    app.run(debug=True)