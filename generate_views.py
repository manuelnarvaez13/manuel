import os
import numpy as np
import trimesh
import pyrender
import cv2
from PIL import Image
from mvmp import Facemarker  # Facemarker è una classe che permette di estrarre i landmark facciali 3D da un modello 3D, essenziale per l'allineamento della mesh.

# =========================================================
# CONFIGURAZIONE GENERALE
# =========================================================
PROJECT_ROOT = r"D:\Admin\Documents\progetto"
SUBJECTS_ROOT = os.path.join(PROJECT_ROOT, "subjects")
IMG_SIZE = 2048

ANGLES = [-90, -60, -45, -30, 0, 30, 45, 60, 90]


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
# 2. ALLINEAMENTO A 2 LIVELLI (PRIMA 5 PUNTI, POI FALLBACK 23 PUNTI)
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
        return mesh

    # ---------------------------------------------------------
    # STRATEGIA 2: FALLBACK SUL METODO CON 23 LANDMARK + DEPTH_OFFSET
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

    return mesh


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
# 4. PIPELINE PRINCIPALE DI RENDERING
# =========================================================
def render_subject_views(subject_folder):
    folder_name = os.path.basename(subject_folder)
    
    # 1. CONTROLLO PRESENZA FILE .BMP
    bmp_files = [f for f in os.listdir(subject_folder) if f.lower().endswith('.bmp')]
    if not bmp_files:
        print(f"⛔ Saltato {folder_name}: Nessun file .bmp trovato.")
        return

    # 2. CONTROLLO PRESENZA FILE .OBJ
    obj_files = [f for f in os.listdir(subject_folder) if f.lower().endswith('.obj')]
    if not obj_files:
        print(f"❌ Nessun file .obj trovato in: {subject_folder}")
        return

    obj_path = os.path.join(subject_folder, obj_files[0])
    print(f"\n🔄 Elaborazione soggetto: {folder_name}")

    # STAGE 1: Controllo Landmark ed Allineamento
    try:
        print("1️⃣ Estrazione landmark ed allineamento Procrustes...")
        mesh = align_mesh_with_facemarker(obj_path)
    except Exception as e:
        print(f"⚠️ Impossibile elaborare {folder_name}: {e}")
        return

    # STAGE 2: Controllo e Applicazione Texture (direttamente dal file .BMP)
    print("2️⃣ Controllo e applicazione texture...")
    bmp_name = bmp_files[0]
    percorso_bmp = os.path.join(subject_folder, bmp_name)

    texture_applicata = False
    if os.path.exists(percorso_bmp):
        try:
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
                    print(f"   ✓ Texture BMP applicata: {percorso_bmp}")
                    texture_applicata = True
        except Exception as e:
            print(f"   ⚠️ Errore durante l'applicazione della texture BMP: {e}")

    if not texture_applicata:
        print(f"⛔ Saltato {folder_name}: Impossibile caricare la texture da {percorso_bmp}.")
        return

    # Normalizzazione e Centratura della Mesh
    max_extent = np.max(mesh.extents)
    if max_extent > 0:
        mesh.vertices *= (1.0 / max_extent)

    center = mesh.bounding_box.centroid
    mesh.vertices -= center

    # STAGE 3: Rendering Offscreen
    print("3️⃣ Generazione viste orizzontali centrate...")
    views_folder = os.path.join(subject_folder, "views")
    os.makedirs(views_folder, exist_ok=True)

    renderer = pyrender.OffscreenRenderer(viewport_width=IMG_SIZE, viewport_height=IMG_SIZE)

    scene = pyrender.Scene(
        bg_color=[240/255, 240/255, 240/255],
        ambient_light=[0.4, 0.4, 0.4]
    )

    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(render_mesh)

    fov_rad = 2.0 * np.arctan(24.0 / (2.0 * 50.0))
    radius = 3.0
    target_point = mesh.bounding_box.centroid

    camera = pyrender.PerspectiveCamera(yfov=fov_rad)
    camera_node = scene.add(camera, pose=np.eye(4))

    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)
    key_light_node = scene.add(key_light, pose=np.eye(4))

    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.5)
    scene.add(fill_light, pose=look_at([0, 5, 0], target_point))

    for idx, yaw_deg in enumerate(ANGLES, start=1):
        yaw_rad = np.radians(yaw_deg)

        cam_x = radius * np.sin(yaw_rad)
        cam_y = 0.0
        cam_z = radius * np.cos(yaw_rad)

        camera_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float64)
        camera_pose = look_at(camera_pos, target=target_point)

        scene.set_pose(camera_node, pose=camera_pose)
        scene.set_pose(key_light_node, pose=camera_pose)

        color, _ = renderer.render(scene)
        img_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        out_filename = os.path.join(views_folder, f"view_{idx:02d}.png")
        cv2.imwrite(out_filename, img_bgr)
        print(f"   -> Salvata {out_filename} (Yaw: {yaw_deg}°)")

    renderer.delete()
    print(f"✨ Viste salvate in 'views/' per: {folder_name}\n")


# =========================================================
# 5. ESECUZIONE BATCH
# =========================================================
if __name__ == "__main__":
    INIZIO = 1
    FINE = 200

    print(f"--- Avvio batch rendering dai soggetti {INIZIO:05d} a {FINE:05d} ---")

    for subject_num in range(INIZIO, FINE + 1):
        folder_name = f"{subject_num:05d}"
        subject_folder = os.path.join(SUBJECTS_ROOT, folder_name)

        if os.path.exists(subject_folder):
            render_subject_views(subject_folder)
        else:
            print(f"⚠️ Cartella '{folder_name}' non trovata.")

    print("\n✨ PROCESSO BATCH TERMINATO!")