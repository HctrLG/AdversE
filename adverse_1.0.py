import tkinter as tk
from tkinter import filedialog, messagebox
import face_recognition
import cv2
import numpy as np
import sqlite3
import os
from PIL import Image, ImageTk
from datetime import datetime

# Crear o conectar a nueva base de datos
conn = sqlite3.connect("faces_reiniciada.db")
cursor = conn.cursor()
cursor.execute("""CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    birthdate TEXT,
    encoding BLOB NOT NULL
)""")
conn.commit()

# Función para registrar una nueva persona
def registrar_persona():
    nombre = entry_nombre.get()
    fecha_nac = entry_fecha.get()

    if not nombre or not fecha_nac:
        messagebox.showerror("Error", "Introduce nombre y fecha de nacimiento.")
        return

    filepath = filedialog.askopenfilename(filetypes=[("Imagen", "*.jpg *.jpeg *.png")])
    if not filepath:
        return

    image = face_recognition.load_image_file(filepath)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        messagebox.showerror("Error", "No se detectó ninguna cara.")
        return

    encoding = encodings[0]
    encoding_bytes = encoding.tobytes()

    cursor.execute("INSERT INTO faces (name, birthdate, encoding) VALUES (?, ?, ?)", (nombre, fecha_nac, encoding_bytes))
    conn.commit()
    messagebox.showinfo("Éxito", f"Persona '{nombre}' registrada correctamente.")

# Función para reconocer rostros en una imagen
def reconocer_rostros():
    filepath = filedialog.askopenfilename(filetypes=[("Imagen", "*.jpg *.jpeg *.png")])
    if not filepath:
        return

    unknown_image = face_recognition.load_image_file(filepath)
    unknown_encodings = face_recognition.face_encodings(unknown_image)
    face_locations = face_recognition.face_locations(unknown_image)

    known_encodings = []
    known_names = []
    known_birthdates = []

    cursor.execute("SELECT name, birthdate, encoding FROM faces")
    for name, birthdate, encoding_blob in cursor.fetchall():
        known_encodings.append(np.frombuffer(encoding_blob, dtype=np.float64))
        known_names.append(name)
        known_birthdates.append(birthdate)

    image_bgr = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

    for (top, right, bottom, left), face_encoding in zip(face_locations, unknown_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if True in matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                nombre = known_names[best_match_index]
                fecha = known_birthdates[best_match_index]
                color = (0, 255, 0)
                texto = f"{nombre} - {fecha}"
            else:
                color = (0, 0, 255)
                texto = "Ninguna coincidencia"
        else:
            color = (0, 0, 255)
            texto = "Ninguna coincidencia"

        cv2.rectangle(image_bgr, (left, top), (right, bottom), color, 2)
        cv2.putText(image_bgr, texto, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow("Resultado", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Interfaz Tkinter
root = tk.Tk()
root.title("Reconocimiento Facial")
root.geometry("400x250")
root.configure(bg="#2e2e2e")  # Fondo gris oscuro

tk.Label(root, text="Nombre:", bg="#2e2e2e", fg="white").pack(pady=(20, 0))
entry_nombre = tk.Entry(root)
entry_nombre.pack()

tk.Label(root, text="Fecha de nacimiento (YYYY-MM-DD):", bg="#2e2e2e", fg="white").pack(pady=(10, 0))
entry_fecha = tk.Entry(root)
entry_fecha.pack()

tk.Button(root, text="Registrar Persona", bg="#4CAF50", fg="white", command=registrar_persona).pack(pady=10)
tk.Button(root, text="Reconocer Rostros", bg="#2196F3", fg="white", command=reconocer_rostros).pack(pady=10)

root.mainloop()
