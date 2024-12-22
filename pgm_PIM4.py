import streamlit as st
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import paramiko
import os

# Email Configuration
SMTP_SERVER = st.secrets["SMTP_SERVER"]
SMTP_PORT = int(st.secrets["SMTP_PORT"])
EMAIL_USER = st.secrets["EMAIL_USER"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]
NOTIFICATION_EMAIL = st.secrets["NOTIFICATION_EMAIL"]

# Remote server configuration
REMOTE_HOST = st.secrets["REMOTE_HOST"]
REMOTE_USER = st.secrets["REMOTE_USER"]
REMOTE_PASSWORD = st.secrets["REMOTE_PASSWORD"]
REMOTE_PORT = int(st.secrets["REMOTE_PORT"])
REMOTE_DIR = st.secrets["REMOTE_DIR"]
REMOTE_FILE = "muestra.fasta"
LOCAL_FILE = "muestra.fasta"

# Custom styles
st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(to bottom, #f4ecd8, #e2d5c3);
            color: #333;
            font-family: Arial, sans-serif;
        }
        .main-title {
            font-size: 3rem;
            color: #333;
            text-align: center;
            margin-top: 1rem;
        }
        .subtitle {
            font-size: 1.5rem;
            color: #555;
            text-align: center;
            margin-bottom: 2rem;
        }
        .upload-area {
            background-color: #ffffff;
            border: 2px dashed #ccc;
            padding: 1rem;
            text-align: center;
            margin-bottom: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .results-title {
            font-size: 1.75rem;
            color: #222;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .email-input {
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

def classify_sequence(sequence):
    cpp_keywords = ["CPP"]
    ncpp_keywords = ["NCPP"]
    unf_keywords = ["UNF"]
    par_keywords = ["PAR"]

    if any(keyword in sequence.upper() for keyword in cpp_keywords):
        return "CPP"
    elif any(keyword in sequence.upper() for keyword in ncpp_keywords):
        return "NCPP"
    elif any(keyword in sequence.upper() for keyword in unf_keywords):
        return "UNF"
    elif any(keyword in sequence.upper() for keyword in par_keywords):
        return "PAR"
    else:
        return "Unknown"

def send_email_with_results(user_name, user_email):
    try:
        with open("totales.txt", "r") as f:
            file_content_totales = f.read()

        with open(LOCAL_FILE, "r") as f:
            file_content_fasta = f.read()

        message = MIMEMultipart()
        message['From'] = EMAIL_USER
        message['To'] = user_email
        message['Cc'] = NOTIFICATION_EMAIL
        message['Subject'] = "Processed Results from Remote Server"

        body = f"""Dear {user_name},

Please find attached the processed results (totales.txt) from the remote server.
Additionally, your uploaded file is also included for reference.

Best regards,
Protein Analysis Team"""
        message.attach(MIMEText(body, 'plain'))

        # Attach totales.txt
        attachment_totales = MIMEText(file_content_totales, 'plain')
        attachment_totales.add_header('Content-Disposition', 'attachment', filename='totales.txt')
        message.attach(attachment_totales)

        # Attach the user's uploaded file
        attachment_fasta = MIMEText(file_content_fasta, 'plain')
        attachment_fasta.add_header('Content-Disposition', 'attachment', filename='muestra.fasta')
        message.attach(attachment_fasta)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(message)
        st.success("Result file sent to the user successfully.")

    except Exception as e:
        st.error(f"Failed to send the result file via email: {e}")

def upload_and_execute_on_remote(user_name, user_email):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(REMOTE_HOST, username=REMOTE_USER, password=REMOTE_PASSWORD, port=REMOTE_PORT)

        sftp = ssh.open_sftp()
        sftp.put(LOCAL_FILE, os.path.join(REMOTE_DIR, REMOTE_FILE))
        st.success("File uploaded to the remote server successfully.")

        stdin, stdout, stderr = ssh.exec_command(f"cd {REMOTE_DIR} && bash ejecucion.sh")
        stdout.channel.recv_exit_status()
        st.success("Remote script executed successfully.")

        sftp.get(os.path.join(REMOTE_DIR, "totales.txt"), "totales.txt")
        st.success("Result file downloaded from the remote server successfully.")

        sftp.close()
        ssh.close()

        if user_email:
            send_email_with_results(user_name, user_email)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Streamlit application
st.markdown("<div class='main-title'>POLARITY INDEX METHOD PIM v3.0</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>PIM v3.0 enables the determination of the affinity (%) of uploaded proteins with different groups of structural proteins.</div>", unsafe_allow_html=True)

# Collect user name and email
st.markdown("<div class='email-input'>", unsafe_allow_html=True)
user_name = st.text_input("Enter your name:")
user_email = st.text_input("Enter your email to receive the results:")
st.markdown("</div>", unsafe_allow_html=True)

# Upload or input protein sequences
st.markdown("<div class='upload-area'>Upload a FASTA file with protein sequences:</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your FASTA file:", type=["fasta"], label_visibility='hidden')

if uploaded_file:
    with open(LOCAL_FILE, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("FASTA file uploaded successfully.")

    if user_name and user_email:
        upload_and_execute_on_remote(user_name, user_email)
    else:
        st.error("Please enter your name and email address.")

st.markdown("""
    <div style='text-align: center; margin-top: 50px; font-size: 0.9rem; color: #555;'>
        &copy; All rights reserved by Carlos Polanco. Mexico City, Mexico. Contact: polanco@unam.mx.
    </div>
""", unsafe_allow_html=True)

