import os
import numpy as np
import trimesh
import pyrender
import cv2
from PIL import Image
from mvmp import Facemarker

# =========================================================
# CONFIGURAZIONE GENERALE
# =========================================================
PROJECT_ROOT = r"D:\Admin\Documents\progetto"
SUBJECTS_ROOT = os.path.join(PROJECT_ROOT, "subjects")
THESIS_OUTPUT_ROOT = r"D:\Admin\Documents\progetto\immagini per tesi"
IMG_SIZE = 2048

# Percorso diretto al singolo soggetto
SUBJECT_ID = "00002"
subject_folder = os.path.join(SUBJECTS_ROOT, SUBJECT_ID)


# =========================================================
# 1. PROCRUSTES RIGIDO (SE(3))
# =========================================================
def procrustes_rigid_transform(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)

    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid

    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_centroid - R @ src_centroid

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


# =========================================================
# 2. ALLINEAMENTO A 2 LIVELLI E RESTITUZIONE MATRICE T
# =========================================================
def align_mesh_with_facemarker(obj_path):
    mesh = trimesh.load(obj_path, force='mesh')

    marker = Facemarker()
    result = marker.predict(obj_path)
    raw_landmarks_3d = result.landmarks_3d

    if raw_landmarks_3d is None or len(raw_landmarks_3d) == 0:
        raise ValueError("Nessun landmark estratto da Facemarker.")

    # ---------------------------------------------------------
    # STRATEGIA 1: TENTATIVO CON I 5 LANDMARK FONDAMENTALI
    # ---------------------------------------------------------
    idx_5_map = {
        'naso': 4,
        'occhio_sx': 33,
        'occhio_dx': 263,
        'fronte': 10,
        'mento': 152
    }

    target_5 = np.array([
        [0.0, 0.0, 0.0],    # naso
        [-1.0, 0.0, 0.0],   # occhio_sx
        [1.0, 0.0, 0.0],    # occhio_dx
        [0.0, 1.0, 0.0],    # fronte
        [0.0, -1.0, 0.0]    # mento
    ], dtype=np.float64)

    source_5 = []
    tutti_e_5_presenti = True

    for nome, idx in idx_5_map.items():
        if idx in raw_landmarks_3d:
            source_5.append(raw_landmarks_3d[idx])
        else:
            tutti_e_5_presenti = False
            break

    if tutti_e_5_presenti and len(source_5) == 5:
        print("   ✓ Allineamento riuscito usando i 5 LANDMARK PRIMARI.")
        T = procrustes_rigid_transform(source_5, target_5)
        mesh.apply_transform(T)
        return mesh, T, "5_primari", raw_landmarks_3d

    # ---------------------------------------------------------
    # STRATEGIA 2: FALLBACK SUL METODO CON 23 LANDMARK
    # ---------------------------------------------------------
    print("   ⚠️ I 5 landmark primari non sono tutti presenti. Inizio FALLBACK a 23 landmark...")

    idx_23_map = {
        'naso_punta': 4, 'naso_radice': 6, 'naso_sotto': 2,
        'occhio_sx_est': 33, 'occhio_sx_int': 133, 'occhio_sx_sopra': 159, 'occhio_sx_sotto': 145,
        'occhio_dx_est': 263, 'occhio_dx_int': 362, 'occhio_dx_sopra': 386, 'occhio_dx_sotto': 374,
        'sopracciglio_sx': 70, 'sopracciglio_dx': 300,
        'bocca_sx': 61, 'bocca_dx': 291, 'bocca_sopra': 0, 'bocca_sotto': 17,
        'fronte': 10, 'mento': 152,
        'zigomo_sx': 234, 'zigomo_dx': 454,
        'mascella_sx': 172, 'mascella_dx': 397
    }

    DEPTH_OFFSET = 0.55  

    canonical_target_23_map = {
        4: [0.0, 0.0, DEPTH_OFFSET],
        6: [0.0, 0.3, DEPTH_OFFSET - 0.1],
        2: [0.0, -0.2, DEPTH_OFFSET - 0.05],
        33: [-1.0, 0.3, DEPTH_OFFSET - 0.2],
        133: [-0.4, 0.3, DEPTH_OFFSET - 0.2],
        159: [-0.7, 0.45, DEPTH_OFFSET - 0.18],
        145: [-0.7, 0.15, DEPTH_OFFSET - 0.18],
        263: [1.0, 0.3, DEPTH_OFFSET - 0.2],
        362: [0.4, 0.3, DEPTH_OFFSET - 0.2],
        386: [0.7, 0.45, DEPTH_OFFSET - 0.18],
        374: [0.7, 0.15, DEPTH_OFFSET - 0.18],
        70: [-0.7, 0.6, DEPTH_OFFSET - 0.15],
        300: [0.7, 0.6, DEPTH_OFFSET - 0.15],
        61: [-0.5, -0.5, DEPTH_OFFSET - 0.1],
        291: [0.5, -0.5, DEPTH_OFFSET - 0.1],
        0: [0.0, -0.4, DEPTH_OFFSET - 0.05],
        17: [0.0, -0.6, DEPTH_OFFSET - 0.05],
        10: [0.0, 1.0, DEPTH_OFFSET - 0.2],
        152: [0.0, -1.0, DEPTH_OFFSET - 0.2],
        234: [-1.2, 0.0, DEPTH_OFFSET - 0.4],
        454: [1.2, 0.0, DEPTH_OFFSET - 0.4],
        172: [-0.9, -0.8, DEPTH_OFFSET - 0.4],
        397: [0.9, -0.8, DEPTH_OFFSET - 0.4]
    }

    source_23 = []
    target_23 = []

    for name, idx in idx_23_map.items():
        if idx in raw_landmarks_3d:
            source_23.append(raw_landmarks_3d[idx])
            target_23.append(canonical_target_23_map[idx])

    if len(source_23) < 3:
        source_23 = []
        target_23 = []
        for idx, pt in raw_landmarks_3d.items():
            if idx in canonical_target_23_map:
                source_23.append(pt)
                target_23.append(canonical_target_23_map[idx])
            if len(source_23) >= 3:
                break

    if len(source_23) < 3:
        raise ValueError(f"Impossibile allineare: trovati solo {len(source_23)} punti su un minimo di 3 richiesti.")

    print(f"   ℹ️ Allineamento completato in FALLBACK con {len(source_23)} landmark.")
    T = procrustes_rigid_transform(source_23, target_23)
    mesh.apply_transform(T)

    return mesh, T, "23_fallback", raw_landmarks_3d


# =========================================================
# 3. UTILITY CAMERA (LOOK_AT)
# =========================================================
def look_at(position, target, up=None):
    if up is None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    position = np.array(position, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    zaxis = position - target
    norm_z = np.linalg.norm(zaxis)
    if norm_z > 0:
        zaxis /= norm_z

    if np.abs(np.dot(zaxis, up)) > 0.99:
        up = np.array([0.0, 0.0, -1.0] if zaxis[1] > 0 else [0.0, 0.0, 1.0])

    xaxis = np.cross(up, zaxis)
    norm_x = np.linalg.norm(xaxis)
    if norm_x > 0:
        xaxis /= norm_x
    else:
        xaxis = np.array([1.0, 0.0, 0.0])

    yaxis = np.cross(zaxis, xaxis)
    norm_y = np.linalg.norm(yaxis)
    if norm_y > 0:
        yaxis /= norm_y

    M = np.eye(4, dtype=np.float64)
    M[:3, 0] = xaxis
    M[:3, 1] = yaxis
    M[:3, 2] = zaxis
    M[:3, 3] = position

    return M


# =========================================================
# 4. FUNZIONE PER CREARE SFERE PER I LANDMARK
# =========================================================
def create_sphere_node(center, radius=0.015, color=[1.0, 0.0, 0.0]):
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    sphere.apply_translation(center)
    sphere.visual.vertex_colors = np.tile(
        np.array(color + [1.0], dtype=np.float32) * 255, 
        (len(sphere.vertices), 1)
    )
    return pyrender.Mesh.from_trimesh(sphere, smooth=True)


# =========================================================
# 5. ESECUZIONE DIRETTA SUL SOGGETTO 00001
# =========================================================
if __name__ == "__main__":
    print(f"--- Avvio elaborazione tesi per il soggetto {SUBJECT_ID} ---")

    if not os.path.exists(subject_folder):
        raise FileNotFoundError(f"⚠️ Cartella non trovata: {subject_folder}")

    # 1. Ricerca file .bmp e .obj
    bmp_files = [f for f in os.listdir(subject_folder) if f.lower().endswith('.bmp')]
    obj_files = [f for f in os.listdir(subject_folder) if f.lower().endswith('.obj')]

    if not bmp_files or not obj_files:
        raise FileNotFoundError("❌ File .bmp o .obj mancanti all'interno della cartella del soggetto.")

    obj_path = os.path.join(subject_folder, obj_files[0])
    percorso_bmp = os.path.join(subject_folder, bmp_files[0])

    # 2. Estrazione landmark ed allineamento
    print("1️⃣ Estrazione landmark ed allineamento Procrustes...")
    mesh, T, modalita, raw_landmarks_3d = align_mesh_with_facemarker(obj_path)

    # 3. Applicazione Texture BMP
    print("2️⃣ Applicazione texture...")
    img_rgba = cv2.imread(percorso_bmp, cv2.IMREAD_UNCHANGED)
    if img_rgba is not None:
        if len(img_rgba.shape) == 3 and img_rgba.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGR2RGB)
        elif len(img_rgba.shape) == 3 and img_rgba.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2RGBA)
        else:
            img_rgb = img_rgba

        texture_image = Image.fromarray(img_rgb)
        material = trimesh.visual.material.SimpleMaterial(image=texture_image)

        if hasattr(mesh.visual, 'uv'):
            mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=material)
            print("   ✓ Texture BMP applicata con successo.")

    # 4. Normalizzazione e Centratura della Mesh
    max_extent = np.max(mesh.extents)
    if max_extent > 0:
        mesh.vertices *= (1.0 / max_extent)

    center = mesh.bounding_box.centroid
    mesh.vertices -= center

    # Trasformazione dei landmark grezzi con la stessa matrice e centratura
    transformed_landmarks = {}
    for idx, pt in raw_landmarks_3d.items():
        pt_hom = np.append(np.asarray(pt, dtype=np.float64), 1.0)
        pt_trans = (T @ pt_hom)[:3]
        if max_extent > 0:
            pt_trans *= (1.0 / max_extent)
        pt_trans -= center
        transformed_landmarks[idx] = pt_trans

    # Cartella di output specifica per il soggetto
    subject_thesis_folder = os.path.join(THESIS_OUTPUT_ROOT, SUBJECT_ID)
    os.makedirs(subject_thesis_folder, exist_ok=True)

    renderer = pyrender.OffscreenRenderer(viewport_width=IMG_SIZE, viewport_height=IMG_SIZE)

    idx_5_map = {'naso': 4, 'occhio_sx': 33, 'occhio_dx': 263, 'fronte': 10, 'mento': 152}
    idx_23_map = {
        'naso_punta': 4, 'naso_radice': 6, 'naso_sotto': 2,
        'occhio_sx_est': 33, 'occhio_sx_int': 133, 'occhio_sx_sopra': 159, 'occhio_sx_sotto': 145,
        'occhio_dx_est': 263, 'occhio_dx_int': 362, 'occhio_dx_sopra': 386, 'occhio_dx_sotto': 374,
        'sopracciglio_sx': 70, 'sopracciglio_dx': 300,
        'bocca_sx': 61, 'bocca_dx': 291, 'bocca_sopra': 0, 'bocca_sotto': 17,
        'fronte': 10, 'mento': 152,
        'zigomo_sx': 234, 'zigomo_dx': 454,
        'mascella_sx': 172, 'mascella_dx': 397
    }

    target_point = mesh.bounding_box.centroid
    camera_pos = np.array([0.0, 0.0, 3.0], dtype=np.float64)
    camera_pose = look_at(camera_pos, target=target_point)
    fov_rad = 2.0 * np.arctan(24.0 / (2.0 * 50.0))

    # ---------------------------------------------------------
    # GENERAZIONE IMMAGINE 1: 5 LANDMARK PRINCIPALI
    # ---------------------------------------------------------
    print("3️⃣ Generazione immagine con i 5 landmark principali...")
    scene_5 = pyrender.Scene(bg_color=[255, 255, 255], ambient_light=[0.5, 0.5, 0.5])
    scene_5.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    
    camera = pyrender.PerspectiveCamera(yfov=fov_rad)
    scene_5.add(camera, pose=camera_pose)
    
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)
    scene_5.add(light, pose=camera_pose)

    for name, idx in idx_5_map.items():
        if idx in transformed_landmarks:
            sphere_mesh = create_sphere_node(transformed_landmarks[idx], radius=0.018, color=[1.0, 0.0, 0.0]) # Rosso
            scene_5.add(sphere_mesh)

    color_5, _ = renderer.render(scene_5)
    out_5_path = os.path.join(subject_thesis_folder, "landmarks_5_view.png")
    cv2.imwrite(out_5_path, cv2.cvtColor(color_5, cv2.COLOR_RGB2BGR))
    print(f"   -> Salvata: {out_5_path}")

    # ---------------------------------------------------------
    # GENERAZIONE IMMAGINE 2: 23 LANDMARK
    # ---------------------------------------------------------
    print("4️⃣ Generazione immagine con i 23 landmark...")
    scene_23 = pyrender.Scene(bg_color=[255, 255, 255], ambient_light=[0.5, 0.5, 0.5])
    scene_23.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    
    scene_23.add(camera, pose=camera_pose)
    scene_23.add(light, pose=camera_pose)

    for name, idx in idx_23_map.items():
        if idx in transformed_landmarks:
            sphere_mesh = create_sphere_node(transformed_landmarks[idx], radius=0.012, color=[0.0, 0.4, 1.0]) # Blu
            scene_23.add(sphere_mesh)

    color_23, _ = renderer.render(scene_23)
    out_23_path = os.path.join(subject_thesis_folder, "landmarks_23_view.png")
    cv2.imwrite(out_23_path, cv2.cvtColor(color_23, cv2.COLOR_RGB2BGR))
    print(f"   -> Salvata: {out_23_path}")

    renderer.delete()
    print(f"\n✨ PROCESSO COMPLETATO! Immagini salvate in: {subject_thesis_folder}")