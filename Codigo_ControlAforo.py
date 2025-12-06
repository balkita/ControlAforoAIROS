import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# ---------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------

# Cargar modelo YOLO (people class = id 0)
model = YOLO("yolov8n.pt")

# Elegir fuente de video
cap = cv2.VideoCapture(0)  # Archivo de video

# Tracking de centroides (historial por objeto)
centroids = {}

# Contadores
total_entries = 0
total_exits = 0

# Historial limitado
MAX_HISTORY = 5


# ---------------------------
# FUNCIÓN PARA OBTENER CENTRO DEL BOUNDING BOX
# ---------------------------

def get_centroid(x1, y1, x2, y2):
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


# ---------------------------
# LOOP PRINCIPAL
# ---------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o error.")
        break

    # Detección YOLO
    results = model.track(frame, persist=True)

    # Frame anotado
    annotated = results[0].plot()

    # Tamaño del frame
    H, W, _ = annotated.shape

    # ---------------------------
    # DEFINICIÓN DE ZONAS
    # ---------------------------

    entry_left = 0
    entry_right = int(W * 0.20)  # 0–20%
    exit_left = int(W * 0.80)    # 80–100%
    exit_right = W

    # Dibujar zonas en pantalla
    cv2.rectangle(annotated, (entry_left, 0), (entry_right, H), (0, 255, 255), 2)
    cv2.putText(annotated, "ENTRY ZONE", (entry_left + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.rectangle(annotated, (exit_left, 0), (exit_right, H), (255, 0, 0), 2)
    cv2.putText(annotated, "EXIT ZONE", (exit_left + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # ---------------------------
    # PROCESAR FOLLOW DE OBJETOS
    # ---------------------------

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])

            # Solo detectar personas
            if cls != 0:
                continue

            if box.id is None:
                continue  # ignorar objetos sin ID

            obj_id = int(box.id[0])

            x1, y1, x2, y2 = box.xyxy[0]

            cx, cy = get_centroid(x1, y1, x2, y2)

            # Guardar historial
            if obj_id not in centroids:
                centroids[obj_id] = deque(maxlen=MAX_HISTORY)

            centroids[obj_id].append(cx)

            # Si no hay suficiente historial, saltar
            if len(centroids[obj_id]) < 2:
                continue

            prev_x = centroids[obj_id][-2]
            direction = cx - prev_x  # + derecha, - izquierda

            # ---------------------------
            # LÓGICA DE ENTRADA Y SALIDA
            # ---------------------------

            # Movimiento hacia la IZQUIERDA ← (entrada)
            if direction < 0:
                # Cruza desde zona neutra hacia 0–20%
                if prev_x > entry_right and cx <= entry_right:
                    total_entries += 1
                    print(f"[ENTRY] ID {obj_id} entro.")

            # Movimiento hacia la DERECHA → (salida)
            elif direction > 0:
                # Cruza desde zona neutra hacia 80–100%
                if prev_x < exit_left and cx >= exit_left:
                    total_exits += 1
                    print(f"[EXIT] ID {obj_id} salio.")

            # Dibujar ID y dirección
            cv2.putText(annotated, f"ID {obj_id}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # ---------------------------
    # MOSTRAR CONTADORES
    # ---------------------------

    cv2.putText(annotated, f"Entradas: {total_entries}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

    cv2.putText(annotated, f"Salidas: {total_exits}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)

    # ---------------------------
    # REDIMENSIONAR PARA EVITAR ZOOM RARO
    # ---------------------------

    annotated = cv2.resize(annotated, (1280, 720))

    # Mostrar
    cv2.imshow("People Counter (Entry/Exit Zones)", annotated)

    # Salir con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
