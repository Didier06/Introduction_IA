import cv2

cap = cv2.VideoCapture("/dev/video0")

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire la webcam.")
        break

    cv2.imshow("Webcam", frame)

    # Quittez avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
