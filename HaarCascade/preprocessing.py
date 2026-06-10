import cv2
import os

# 1. Tentukan nama folder sumber (berisi VIDEO)
vid_pos_dir = '../FixedDataset/Yawn'
vid_neg_dir = '../FixedDataset/Not'

# 2. Tentukan nama folder tujuan (berisi GAMBAR hasil ekstrak)
out_pos_dir = '../YOLO/positive'
out_neg_dir = '../YOLO/negative'

os.makedirs(out_pos_dir, exist_ok=True)
os.makedirs(out_neg_dir, exist_ok=True)

# Load detektor wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Jeda frame (Ambil 1 gambar setiap 10 frame video agar data bervariasi)
FRAME_SKIP = 10 

print("--- Memulai Ekstraksi Video POSITIF ---")
pos_count = 1

for filename in os.listdir(vid_pos_dir):
    if filename.endswith(('.mp4')):
        filepath = os.path.join(vid_pos_dir, filename)
        cap = cv2.VideoCapture(filepath)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break # Video habis
            
            # Hanya proses frame sesuai jeda (frame ke-0, 10, 20, dst)
            if frame_idx % FRAME_SKIP == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                
                for (x, y, w, h) in faces:
                    # Ambil 1/3 bagian bawah wajah untuk mulut
                    mulut_y = y + int(h * 0.6)
                    mulut_h = int(h * 0.4)
                    mulut_x = x + int(w * 0.15)
                    mulut_w = int(w * 0.7)
                    
                    if mulut_y + mulut_h <= frame.shape[0] and mulut_x + mulut_w <= frame.shape[1]:
                        crop_mulut = frame[mulut_y:mulut_y+mulut_h, mulut_x:mulut_x+mulut_w]
                        crop_mulut_resized = cv2.resize(crop_mulut, (24, 24))
                        
                        nama_file_baru = f"yawn_{pos_count}.jpg"
                        cv2.imwrite(os.path.join(out_pos_dir, nama_file_baru), crop_mulut_resized)
                        pos_count += 1
                    break # Ambil 1 wajah saja per frame
            frame_idx += 1
        cap.release()

print(f"Selesai! Berhasil mengekstrak dan memotong {pos_count - 1} gambar positif.")


print("\n--- Memulai Ekstraksi Video NEGATIF ---")
neg_count = 1

for filename in os.listdir(vid_neg_dir):
    if filename.endswith(('.mp4')):
        filepath = os.path.join(vid_neg_dir, filename)
        cap = cv2.VideoCapture(filepath)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break # Video habis
            
            # Sama, ambil setiap kelipatan FRAME_SKIP
            if frame_idx % FRAME_SKIP == 0:
                # Resize frame agar tidak memakan RAM berlebih
                tinggi_asli, lebar_asli = frame.shape[:2]
                lebar_baru = 500
                tinggi_baru = int((lebar_baru / lebar_asli) * tinggi_asli)
                
                frame_resized = cv2.resize(frame, (lebar_baru, tinggi_baru))
                
                nama_file_baru = f"normal_{neg_count}.jpg"
                cv2.imwrite(os.path.join(out_neg_dir, nama_file_baru), frame_resized)
                neg_count += 1
                
            frame_idx += 1
        cap.release()

print(f"Selesai! Berhasil mengekstrak {neg_count - 1} gambar negatif.")

