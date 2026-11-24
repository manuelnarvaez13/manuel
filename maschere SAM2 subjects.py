import os
import cv2
import mediapipe as mp
import numpy as np   
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# =========================================================
# CONFIG GPU
# =========================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device: {device}")


# =========================================================
# ROOT CARTELLE SOGGETTI
# =========================================================
subjects_root = r"D:\Admin\Documents\progetto\subjects"


# =========================================================
# LOAD MODELLI SAM2
# =========================================================
def load_sam2(model_name):

    base_path = r"D:\Admin\Documents\sam2"

    checkpoints = {
        "tiny": f"{base_path}\\checkpoints\\sam2.1_hiera_tiny.pt",
        "small": f"{base_path}\\checkpoints\\sam2.1_hiera_small.pt",
        "base_plus": f"{base_path}\\checkpoints\\sam2.1_hiera_base_plus.pt",
        "large": f"{base_path}\\checkpoints\\sam2.1_hiera_large.pt"
    }

    configs = {
        "tiny": f"{base_path}\\sam2\\configs\\sam2.1\\sam2.1_hiera_t.yaml",
        "small": f"{base_path}\\sam2\\configs\\sam2.1\\sam2.1_hiera_s.yaml",
        "base_plus": f"{base_path}\\sam2\\configs\\sam2.1\\sam2.1_hiera_b+.yaml",
        "large": f"{base_path}\\sam2\\configs\\sam2.1\\sam2.1_hiera_l.yaml"
    }

    model = build_sam2(
        configs[model_name],
        checkpoints[model_name],
        device=device
    )

    return SAM2ImagePredictor(model)


# =========================================================
# MEDIAPIPE FACE MESH
# =========================================================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


# =========================================================
# REFINEMENT MASCHERA
# =========================================================
def refine_mask(mask):
    kernel = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.GaussianBlur(
        mask,
        (15, 15),
        0
    )

    return mask


# =========================================================
# MODELLI DA USARE
# =========================================================
models = ["tiny", "small", "base_plus", "large"]


# =========================================================
# CARICAMENTO MODELLI UNA VOLTA SOLA
# =========================================================
predictors = {}

for m in models:
    print(f"📦 Caricamento modello {m}...")
    predictors[m] = load_sam2(m)

print("✅ Tutti i modelli caricati!\n")


# =========================================================
# ESTENSIONI VALIDE
# =========================================================
valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# =========================================================
# CICLO DA 00001 A 00200
# =========================================================
# Modificato: Forziamo il ciclo numerico esatto da 1 a 200 come nell'altro script
for num in range(1, 201):
    
    # Genera il nome della cartella con 5 cifre (es. "00001", "00042", "00200")
    subject_folder = f"{num:05d}"

    subject_path = os.path.join(
        subjects_root,
        subject_folder
    )

    # Verifichiamo se la cartella del soggetto esiste sul disco
    if not os.path.exists(subject_path):
        print(f"Cartella {subject_folder} non trovata, passo alla successiva...")
        continue

    # La cartella da cui leggere i render del volto
    views_path = os.path.join(subject_path, "views")

    print("\n====================================================")
    print(f"📂 Elaborazione soggetto: {subject_folder}")
    print(f"📂 Lettura immagini da: {views_path}")
    print("====================================================")

    # Se la cartella 'views' non esiste per questo soggetto, la saltiamo
    if not os.path.exists(views_path):
        print("⚠️ Cartella 'views' non trovata per questo soggetto. Salto.")
        continue

    # =====================================================
    # FILE IMMAGINI DENTRO LA CARTELLA VIEWS
    # =====================================================
    image_files = sorted([
        f for f in os.listdir(views_path)
        if (
            f.lower().endswith(valid_extensions)
            and f.lower().startswith("view")
        )
    ])

    if not image_files:
        print("⚠️ Nessuna immagine trouvata nella cartella views.")
        continue

    # =====================================================
    # LOOP IMMAGINI
    # =====================================================
    for img_name in image_files:

        print(f"\n📷 Processing: {img_name}")

        image_path = os.path.join(
            views_path,
            img_name
        )

        image = cv2.imread(image_path)

        if image is None:
            print("❌ Errore caricamento.")
            continue

        h, w = image.shape[:2]

        # =================================================
        # RESIZE
        # spiegare perche si fa il risize: per uniformare le dimensioni delle immagini e ridurre il carico computazionale durante l'elaborazione.
        # =================================================
        new_width = 600
        aspect_ratio = new_width / w
        new_height = int(h * aspect_ratio)

        img = cv2.resize(
            image,
            (new_width, new_height)
        )

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # =================================================
        # FACE LANDMARKS
        # =================================================
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            print("❌ Nessun volto rilevato.")
            continue

        points = []

        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                x = int(lm.x * new_width)
                y = int(lm.y * new_height)
                points.append([x, y])

        points = np.array(points)

        # =================================================
        # HULL
        # =================================================
        hull = cv2.convexHull(points)

        # =================================================
        # BOX
        # =================================================
        hull_x = hull[:, :, 0]
        hull_y = hull[:, :, 1]
        x_min = hull_x.min()
        x_max = hull_x.max()
        y_min = hull_y.min()
        y_max = hull_y.max()

        margin = 20

        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(new_width, x_max + margin)
        y_max = min(new_height, y_max + margin)

        box = np.array([
            x_min,
            y_min,
            x_max,
            y_max
        ])

        # =================================================
        # PUNTI POSITIVI
        # =================================================
        center = np.array([
            [int((x_min + x_max) / 2),
             int((y_min + y_max) / 2)]
        ])

        hull_points = hull.reshape(-1, 2)[::20]

        # =================================================
        # PUNTI NEGATIVI
        # =================================================
        neg_points = np.array([
            (x_min - 10, y_min - 10),
            (x_max + 10, y_min - 10),
            (x_min - 10, y_max + 10),
            (x_max + 10, y_max + 10)
        ])

        neg_points = np.clip(
            neg_points,
            [0, 0],
            [new_width - 1, new_height - 1]
        )

        # =================================================
        # COMBINAZIONE
        # =================================================
        point_coords = np.vstack([
            center,
            hull_points,
            neg_points
        ])

        point_labels = np.array(
            [1] * (len(center) + len(hull_points))
            + [0] * len(neg_points)
        )

        base_name = os.path.splitext(img_name)[0]

        # =================================================
        # LOOP MODELLI
        # =================================================
        for m in models:

            print(f"   → {m}")

            # L'output viene salvato fuori da 'views', direttamente in 'subject_path'
            model_output_dir = os.path.join(
                subject_path,
                f"masks_{m}"
            )

            os.makedirs(
                model_output_dir,
                exist_ok=True
            )

            mask_output_path = os.path.join(
                model_output_dir,
                f"{base_name}_mask_SAM2.png"
            )

            # Skip se già esiste
            if os.path.exists(mask_output_path):
                print("      ⏩ Già elaborato.")
                continue

            predictor = predictors[m]

            predictor.set_image(img)

            with torch.inference_mode():

                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=False
                )

            mask = masks[0].astype(np.uint8) * 255

            # =============================================
            # REFINEMENT
            # =============================================
            refined = refine_mask(mask)

            _, refined_bin = cv2.threshold(
                refined,
                127,
                255,
                cv2.THRESH_BINARY
            )

            # =============================================
            # SALVATAGGIO SOLO MASCHERA BINARIA
            # =============================================
            cv2.imwrite(
                mask_output_path,
                refined_bin
            )

        # =================================================
        # PULIZIA CACHE
        # =================================================
        if device == "cuda":
            torch.cuda.empty_cache()


# =========================================================
# FINE
# =========================================================
print("\n🎉 Elaborazione di tutti i soggetti (fino a 00200) completata!")