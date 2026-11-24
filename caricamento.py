import os
import trimesh
import cv2
from PIL import Image
import numpy as np
import pyrender

# =========================================================
# CONFIGURAZIONE PERCORSI E RISOLUZIONE
# =========================================================
obj_path = r"D:\Admin\Documents\progetto\subjects\00001\130925161618.obj"
folder = os.path.dirname(obj_path)

output_dir = r"D:\Admin\Documents\progetto\immagini per tesi"
os.makedirs(output_dir, exist_ok=True)

IMG_RES = 3000  # Risoluzione più alta per la tesi

# Carica il modello 3D
mesh = trimesh.load(obj_path, force='mesh')

# Cerca automaticamente un file immagine (PNG o JPG) nella cartella
image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.bmp'))]

if image_files:
    texture_path = os.path.join(folder, image_files[0])
    print(f"Trovata texture: {texture_path}")
    
    # Carica e applica la texture
    img_rgba = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
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
            print("Texture applicata con successo!")

# =========================================================
# SALVATAGGIO IMMAGINE AD ALTA RISOLUZIONE E SENZA SPAZIO BIANCO
# =========================================================
print("Generazione e salvataggio dell'immagine ad alta risoluzione...")
try:
    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene = pyrender.Scene(bg_color=[255, 255, 255], ambient_light=[0.5, 0.5, 0.5])
    scene.add(render_mesh)

    # Calcolo del centro esatto e della scala del modello per avvicinare la camera
    center = mesh.bounding_box.centroid
    extents = mesh.extents
    max_extent = np.max(extents)

    # Telecamera ortografica o prospettica ravvicinata per ridurre lo spazio bianco
    # Riducendo il fattore di distanza (da 2.5 a circa 1.4-1.6) il soggetto riempirà di più l'inquadratura
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = center + np.array([0, 0, max_extent * 1.5]) 
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Rendering ad alta risoluzione
    renderer = pyrender.OffscreenRenderer(viewport_width=IMG_RES, viewport_height=IMG_RES)
    color, _ = renderer.render(scene)
    renderer.delete()

    # Salva il file immagine
    base_name = os.path.splitext(os.path.basename(obj_path))[0]
    out_image_path = os.path.join(output_dir, f"{base_name}_tesi.png")
    
    img_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_image_path, img_bgr)
    print(f"Immagine ad alta risoluzione salvata con successo in: {out_image_path}")

except Exception as e:
    print(f"Errore durante il salvataggio dell'immagine: {e}")

# =========================================================
# MOSTRA LA FINESTRA 3D INTERATTIVA
# =========================================================
print("Apertura finestra 3D interattiva...")
mesh.show()