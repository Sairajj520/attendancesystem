import os
import json
import cv2
import numpy as np
import face_recognition
import mysql.connector
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request

faces_folder = 'faces/'
embeddings_file = 'face_embeddings.json'

app = Flask(__name__)

response_message = {"message": ""}
camera = None

def store_face_embeddings():
    face_data = {}
    for file in os.listdir(faces_folder):
        if file.endswith('.jpg') or file.endswith('.png'):
            img_path = os.path.join(faces_folder, file)
            img = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(img)
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                face_id = int(os.path.splitext(file)[0])
                face_data[face_id] = {'embedding': face_encoding.tolist()}
    with open(embeddings_file, 'w') as f:
        json.dump(face_data, f)
    print("Face embeddings stored successfully.")

def load_face_embeddings():
    if not os.path.exists(embeddings_file):
        return {}
    with open(embeddings_file, 'r') as f:
        face_data = json.load(f)
    return face_data

def connect_db():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="admin",
        database="attendance_db"
    )
    return connection

def update_attendance(face_id):
    connection = connect_db()
    cursor = connection.cursor()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    query = "UPDATE attendance SET time = %s WHERE id = %s"
    cursor.execute(query, (current_time, face_id))
    connection.commit()
    cursor.close()
    connection.close()

def compare_face_embeddings(detected_embedding, stored_embeddings):
    detected_embedding = np.array(detected_embedding)
    for face_id, face_info in stored_embeddings.items():
        stored_embedding = np.array(face_info['embedding'])
        results = face_recognition.compare_faces([stored_embedding], detected_embedding)
        if results[0]:
            return face_id
    return None

def gen_frames():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Camera could not be opened.")
            return

    stored_embeddings = load_face_embeddings()
    start_time = datetime.now()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_frame)
            face_locations = face_recognition.face_locations(rgb_frame)

            # If no face detected for 30 seconds, alert and reset
            if (datetime.now() - start_time).seconds > 30:
                update_response_message('No face detected')
                break

            # If face is detected, check for match
            if face_encodings:
                for face_encoding in face_encodings:
                    matched_face_id = compare_face_embeddings(face_encoding, stored_embeddings)
                    if matched_face_id:
                        update_attendance(matched_face_id)
                        update_response_message('Attendance marked')
                        break
                else:
                    # If no match is found, alert unrecognized face
                    update_response_message('Unrecognized face')
                    break

            # Draw rectangle on detected faces
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        if camera:
            camera.release()  # Ensure camera is released properly after alert or exit

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera
    if camera:
        camera.release()  # Ensure the camera is released before starting a new session
    camera = None  # Reset camera
    return jsonify({'status': 'camera_started'})

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    global response_message
    stored_embeddings = load_face_embeddings()
    detected_faces = []  # Placeholder for detected face encodings

    if len(detected_faces) == 0:
        update_response_message('No face detected')
        return jsonify({'message': 'No face detected'}), 400

    face_matched = False
    for detected_face in detected_faces:
        matched_face_id = compare_face_embeddings(detected_face, stored_embeddings)
        if matched_face_id:
            update_attendance(matched_face_id)
            face_matched = True
            break

    if face_matched:
        update_response_message('Attendance marked')
        return jsonify({'message': 'Attendance marked'}), 200

    update_response_message('Unrecognized face')
    return jsonify({'message': 'Unrecognized face'}), 400

@app.route('/register-user', methods=['POST'])
def register_user():
    global response_message
    user_id = request.form['id']
    user_name = request.form['name']
    user_photo = request.files['photo']

    connection = connect_db()
    cursor = connection.cursor()

    # Check if ID already exists
    cursor.execute('SELECT * FROM attendance WHERE id = %s', (user_id,))
    result = cursor.fetchone()

    if result:
        message = 'Already registered'
    else:
        photo_path = os.path.join(faces_folder, f'{user_id}.jpg')
        user_photo.save(photo_path)
        try:
            cursor.execute('INSERT INTO attendance (id, name, time) VALUES (%s, %s, NOW())', (user_id, user_name))
            connection.commit()
            message = 'Registration successful'
        except mysql.connector.Error as err:
            message = f'Error: {err}'

    cursor.close()
    connection.close()

    if message == 'Registration successful':
        store_face_embeddings()

    update_response_message(message)
    return jsonify({'message': message}), 200

@app.route('/response_message', methods=['GET'])
def response_message_endpoint():
    global response_message
    return jsonify(response_message)

@app.route('/reset_feed', methods=['POST'])
def reset_feed():
    global response_message, camera
    response_message = {"message": ""}  # Reset the message
    if camera is not None:
        camera.release()  # Release the camera to reset its state
        camera = None  # Reset camera instance
    return jsonify({"status": "reset"})

def update_response_message(message):
    global response_message
    response_message = {"message": message}

@app.route('/clear_response_message', methods=['POST'])
def clear_response_message():
    global response_message
    response_message = {"message": ""}
    return jsonify({'status': 'message_cleared'})


if __name__ == '__main__':
    store_face_embeddings()
    app.run(debug=True)