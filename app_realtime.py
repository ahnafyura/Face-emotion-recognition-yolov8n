import cv2
import dlib
from ultralytics import YOLO
from deepface import DeepFace
import torch 

if torch.cuda.is_available():
    print(f"GPU terdeteksi: {torch.cuda.get_device_name(0)}. Akselerasi aktif.")
else:
    print("PERINGATAN: GPU tidak terdeteksi. Kode akan berjalan lambat di CPU.")

face_detector = YOLO('yolov8n-face-lindevs.pt') 

landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("Semua model (YOLO, dlib) berhasil dimuat.")

cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print(f"Resolusi webcam diatur ke: {cap.get(3)}x{cap.get(4)}")

frame_skip_rate = 2
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % frame_skip_rate != 0:
        if 'annotated_frame' in locals():
            cv2.imshow('Aplikasi Deteksi Wajah Terpadu', annotated_frame)
        else:
            cv2.imshow('Aplikasi Deteksi Wajah Terpadu', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = face_detector(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            face_crop = frame[y1:y2, x1:x2]
            dlib_rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
            landmarks = landmark_predictor(gray_frame, dlib_rect)
            
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
            if face_crop.size != 0:
                try:
                    analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                    dominant_emotion = analysis[0]['dominant_emotion']
                    cv2.putText(frame, dominant_emotion, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                except Exception as e:
                    pass

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    annotated_frame = frame.copy()

    cv2.imshow('Aplikasi Deteksi Wajah Terpadu', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
