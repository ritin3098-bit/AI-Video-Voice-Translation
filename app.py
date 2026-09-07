from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for, session
from flask_cors import CORS
import os
import uuid
import sqlite3
import subprocess
import secrets
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import video_processor

app = Flask(__name__)
# FIX #12: Use env var for secret key instead of hardcoded string
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-change-in-production')
CORS(app, supports_credentials=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'outputs')
app.config['WORK_FOLDER'] = os.path.join(BASE_DIR, 'work')
# FIX #13: Enforce max upload size (500MB)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['WORK_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

DATABASE = os.path.join(BASE_DIR, 'users.db')

def get_db():
    conn = sqlite3.connect(DATABASE, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn

RESET_TOKEN_EXPIRY_MINUTES = 30

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                name TEXT,
                password_hash TEXT NOT NULL
            )
        ''')
        db.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                job_id TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                target_language TEXT NOT NULL,
                output_video TEXT NOT NULL,
                output_srt TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        db.execute('''
            CREATE TABLE IF NOT EXISTS password_resets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT UNIQUE NOT NULL,
                expires_at TEXT NOT NULL,
                used INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # Migration: older DBs created before the 'name' column existed
        existing_cols = [row['name'] for row in db.execute('PRAGMA table_info(users)').fetchall()]
        if 'name' not in existing_cols:
            db.execute('ALTER TABLE users ADD COLUMN name TEXT')

        db.commit()
        db.close()

init_db()


def generate_reset_token(user_id):
    """Creates a single-use, time-limited password reset token for a user."""
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.utcnow() + timedelta(minutes=RESET_TOKEN_EXPIRY_MINUTES)).isoformat()
    db = get_db()
    try:
        db.execute(
            'INSERT INTO password_resets (user_id, token, expires_at, used) VALUES (?, ?, ?, 0)',
            (user_id, token, expires_at)
        )
        db.commit()
    finally:
        db.close()
    return token


def send_reset_email(to_email, reset_link):
    """Sends the reset link via SMTP if configured (SMTP_HOST env var),
    otherwise logs it to the server console so it can be used during local/dev setups
    that don't have email configured yet."""
    smtp_host = os.environ.get('SMTP_HOST')

    if not smtp_host:
        print('\n[VoxBridge] Password reset requested for: {}'.format(to_email))
        print('[VoxBridge] SMTP is not configured (set SMTP_HOST to enable real emails).')
        print('[VoxBridge] Reset link: {}\n'.format(reset_link))
        return False

    smtp_port = int(os.environ.get('SMTP_PORT', 587))
    smtp_user = os.environ.get('SMTP_USER')
    smtp_password = os.environ.get('SMTP_PASSWORD')
    smtp_from = os.environ.get('SMTP_FROM', smtp_user or 'noreply@voxbridge.app')

    body = (
        "Hi,\n\n"
        "We received a request to reset your VoxBridge password.\n\n"
        "Click the link below to set a new password (this link expires in {} minutes):\n{}\n\n"
        "If you didn't request this, you can safely ignore this email."
    ).format(RESET_TOKEN_EXPIRY_MINUTES, reset_link)

    msg = MIMEText(body)
    msg['Subject'] = 'Reset your VoxBridge password'
    msg['From'] = smtp_from
    msg['To'] = to_email

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.sendmail(smtp_from, [to_email], msg.as_string())
        return True
    except Exception as e:
        print('[VoxBridge] Failed to send reset email: {}'.format(e))
        print('[VoxBridge] Reset link (email send failed, use this instead): {}'.format(reset_link))
        return False

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.before_request
def require_login():
    allowed_routes = [
        'login_page', 'register_page', 'api_login', 'api_register', 'static',
        'forgot_password_page', 'api_forgot_password',
        'reset_password_page', 'api_reset_password'
    ]
    if request.endpoint not in allowed_routes and 'user_id' not in session:
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Unauthorized'}), 401
        return redirect(url_for('login_page'))

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        data = request.json
        # FIX: login.html sends 'email', not 'username' — accept both
        username = data.get('username') or data.get('email')
        password = data.get('password')

        db = get_db()
        try:
            user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        finally:
            db.close()

        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['name'] = user['name']
            session.modified = True
            # FIX: return the user's display name so the frontend's
            # "Welcome back, {name}!" message actually shows something
            return jsonify({'success': True, 'name': user['name'] or user['username']})
        return jsonify({'error': 'Invalid credentials'}), 401
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        data = request.json
        name = data.get('name')
        username = data.get('username') or data.get('email')
        password = data.get('password')

        db = get_db()
        try:
            # FIX: 'name' was being silently dropped before — now saved
            db.execute("INSERT INTO users (username, name, password_hash) VALUES (?, ?, ?)",
                       (username, name, generate_password_hash(password)))
            db.commit()
            return jsonify({'success': True, 'name': name or username})
        except sqlite3.IntegrityError:
            return jsonify({'error': 'Username already exists'}), 400
        finally:
            db.close()
    return render_template('register.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    return login_page()

@app.route('/api/register', methods=['POST'])
def api_register():
    return register_page()

# FIX #1: Add /api/me endpoint so frontend can load user info
@app.route('/api/me')
def api_me():
    # FIX: return the actual name instead of the email/username
    return jsonify({'name': session.get('name') or session.get('username')})

# FIX #2: Add /api/logout endpoint so logout button works
@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'success': True})

# ---------------- FORGOT / RESET PASSWORD ---------------- #

@app.route('/forgot-password', methods=['GET'])
def forgot_password_page():
    return render_template('forgot-password.html')

@app.route('/api/forgot-password', methods=['POST'])
def api_forgot_password():
    data = request.json or {}
    email = (data.get('email') or '').strip()

    # Always return the same generic message whether or not the account exists,
    # so this endpoint can't be used to discover which emails are registered.
    generic_response = jsonify({
        'success': True,
        'message': "If an account exists for that email, we've sent password reset instructions."
    })

    if not email:
        return generic_response

    db = get_db()
    try:
        user = db.execute('SELECT * FROM users WHERE username = ?', (email,)).fetchone()
    finally:
        db.close()

    if user:
        token = generate_reset_token(user['id'])
        reset_link = '{}reset-password/{}'.format(request.host_url, token)
        send_reset_email(user['username'], reset_link)

    return generic_response

@app.route('/reset-password/<token>', methods=['GET'])
def reset_password_page(token):
    return render_template('reset-password.html', token=token)

@app.route('/api/reset-password', methods=['POST'])
def api_reset_password():
    data = request.json or {}
    token = data.get('token')
    new_password = data.get('password')

    if not token or not new_password:
        return jsonify({'error': 'Missing token or password.'}), 400
    if len(new_password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters.'}), 400

    db = get_db()
    try:
        reset_row = db.execute(
            'SELECT * FROM password_resets WHERE token = ?', (token,)
        ).fetchone()

        if not reset_row or reset_row['used']:
            return jsonify({'error': 'This reset link is invalid or has already been used.'}), 400

        if datetime.utcnow() > datetime.fromisoformat(reset_row['expires_at']):
            return jsonify({'error': 'This reset link has expired. Please request a new one.'}), 400

        db.execute('UPDATE users SET password_hash = ? WHERE id = ?',
                   (generate_password_hash(new_password), reset_row['user_id']))
        db.execute('UPDATE password_resets SET used = 1 WHERE id = ?', (reset_row['id'],))
        db.commit()
    finally:
        db.close()

    return jsonify({'success': True})

@app.route('/')
@app.route('/dashboard')
def index_page():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # FIX #14: Validate file extension server-side
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Unsupported file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    job_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)

    job_dir = os.path.join(app.config['WORK_FOLDER'], job_id)
    os.makedirs(job_dir, exist_ok=True)

    video_path = os.path.join(job_dir, filename)
    file.save(video_path)

    audio_path = os.path.join(job_dir, 'audio.wav')

    # FIX #10: Catch ffmpeg errors on extract_audio (e.g. video has no audio track)
    try:
        video_processor.extract_audio(video_path, audio_path)
    except subprocess.CalledProcessError:
        return jsonify({'error': 'Could not extract audio. Make sure the video has an audio track.'}), 422

    segments = video_processor.transcribe_audio(audio_path)

    return jsonify({
        'job_id': job_id,
        'video_filename': filename,
        'segments': segments
    })

@app.route('/api/video/<job_id>/<filename>')
def serve_video(job_id, filename):
    video_path = os.path.join(app.config['WORK_FOLDER'], job_id, secure_filename(filename))
    return send_file(video_path)

@app.route('/api/audio/<job_id>')
def serve_audio(job_id):
    audio_path = os.path.join(app.config['WORK_FOLDER'], job_id, 'audio.wav')
    return send_file(audio_path)

@app.route('/api/translate', methods=['POST'])
def api_translate():
    data = request.json
    segments = data.get('segments', [])
    target_language = data.get('target_language', 'hi')
    translated = video_processor.translate_segments(segments, target_language)
    return jsonify({'segments': translated})

@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    job_id = data.get('jobId')
    video_filename = data.get('videoFilename')
    segments = data.get('segments')
    target_language = data.get('target_language', 'hi')
    tts_engine = data.get('tts_engine', 'edge')
    lip_sync = data.get('lip_sync', False)

    job_dir = os.path.join(app.config['WORK_FOLDER'], job_id)
    video_path = os.path.join(job_dir, secure_filename(video_filename))

    output_video_filename = f"final_{job_id}.mp4"
    output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], output_video_filename)

    output_srt_filename = f"subtitles_{job_id}.srt"
    output_srt_path = os.path.join(app.config['OUTPUT_FOLDER'], output_srt_filename)

    # FIX #11: Catch processing errors so client gets a proper error response
    try:
        video_processor.merge_audio_video(
            video_path, segments, target_language, output_video_path, job_dir,
            tts_engine=tts_engine, lip_sync=lip_sync
        )
        video_processor.generate_srt(segments, output_srt_path)
    except subprocess.CalledProcessError as e:
        return jsonify({'error': 'Video processing failed. Check ffmpeg logs.'}), 500

    if not os.path.exists(output_video_path):
        return jsonify({'error': 'Output video was not created.'}), 500

    # Save completed job to DB so history and stats work
    db = get_db()
    try:
        db.execute(
            'INSERT INTO jobs (user_id, job_id, original_filename, target_language, output_video, output_srt) VALUES (?, ?, ?, ?, ?, ?)',
            (session['user_id'], job_id, video_filename, target_language, output_video_filename, output_srt_filename)
        )
        db.commit()
    finally:
        db.close()

    return jsonify({
        'output_video': output_video_filename,
        'output_srt': output_srt_filename
    })

@app.route('/api/download/<filename>')
def api_download(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], secure_filename(filename))
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


@app.route('/api/history')
def api_history():
    db = get_db()
    try:
        rows = db.execute(
            'SELECT * FROM jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT 20',
            (session['user_id'],)
        ).fetchall()
    finally:
        db.close()

    LANGUAGE_NAMES = {
        'hi': 'Hindi', 'mr': 'Marathi', 'ta': 'Tamil',
        'te': 'Telugu', 'bn': 'Bengali', 'gu': 'Gujarati'
    }

    jobs = [{
        'job_id': r['job_id'],
        'original_filename': r['original_filename'],
        'target_language': r['target_language'],
        'language_name': LANGUAGE_NAMES.get(r['target_language'], r['target_language']),
        'output_video': r['output_video'],
        'output_srt': r['output_srt'],
        'created_at': r['created_at']
    } for r in rows]

    return jsonify({'jobs': jobs, 'total': len(jobs)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)