from deepface.modules import representation, detection, verification
import cv2
import threading

# ==== PREPROCESS FUNCTION ====
def preprocess(path):
    if isinstance(path, str):
        img = cv2.imread(path)
    else:
        img = path

    faces = detection.detect_faces(detector_backend="mtcnn", img=img)
    if not faces:
        return None, None  # không có mặt

    face_img = faces[0].img
    face_bb = faces[0].facial_area  # dict với x, y, w, h
    embedding_img = representation.represent(face_img, detector_backend="skip")[0]["embedding"]
    return embedding_img, face_bb

# ==== THREADED PREPROCESS ====
embedding_result = None
processing = False
face_bb = None

def async_preprocess(frame):
    global embedding_result, processing, face_bb
    processing = True
    embedding_result, face_bb = preprocess(frame)
    processing = False

# ==== MAIN ====
cap = cv2.VideoCapture(0)
frame_id = 0
frame_interval = 30

threshold = verification.find_threshold("VGG-Face", "euclidean")
k_image_dir = r"D:\Python plus\AI_For_CV\script\face_recognition\images\my-avatar.jpg"
embedding_k_img, _ = preprocess(k_image_dir)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    if frame_id % frame_interval == 0 and not processing:
        threading.Thread(target=async_preprocess, args=(frame.copy(),)).start()

    # Nếu có embedding mới, dùng và vẽ kết quả
    if embedding_result is not None:
        result = verification.find_distance(embedding_k_img, embedding_result, distance_metric="euclidean")
        check = result <= threshold

        # Hiển thị match / no match
        color = (0, 255, 0) if check else (0, 0, 255)
        cv2.putText(frame, f"Match: {check}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, "Distance: {:0.4f}".format(result), (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Vẽ bounding box nếu có
        if face_bb:
            x, y, w, h = face_bb.x, face_bb.y, face_bb.w, face_bb.h
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("img", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
