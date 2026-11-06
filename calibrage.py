# TP calibrage caméra 

import numpy as np
import cv2
import glob


# Nombre d'intersections INTÉRIEURES (colonnes, lignes)
# Nous avons un damier de 8x8 cases donc on a 7x7 intersections internes.
DAMIER = (7, 7) 
# Taille réelle d'un côté de carré en millimètres
SQUARE_SIZE = 12.0    

# Critères d'arrêt pour l'affinage sous-pixel (précision max)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Préparation des points 3D du monde réel
objp = np.zeros((DAMIER[0] * DAMIER[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:DAMIER[0], 0:DAMIER[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# Stockage des points de toutes les images
objpoints = [] # Points 3D (monde)
imgpoints = [] # Points 2D (image)

# Parcours des images 
images = glob.glob('/Users/MariaAydin1/Documents/M2_VMI_26_27/semestre1/sequence_video/tp1/bdd4/*.jpeg')
print(f"Images chargées : {len(images)}")
gray_shape = None

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray_shape is None:
        gray_shape = gray.shape[::-1] # (largeur, hauteur)

    # Détection des coins
    ret, corners = cv2.findChessboardCorners(gray, DAMIER, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    print(f"Détection sur {fname} : {ret}")

    if ret == True:
        objpoints.append(objp)

        # Affinage de la position des coins pour la précision de calibration
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Affichage de contrôle (optionnel, appuie sur une touche si ça bloque)
        cv2.drawChessboardCorners(img, DAMIER, corners2, ret)
        cv2.imshow('Detection des coins', img)
        cv2.waitKey(0) 

cv2.destroyAllWindows()

# Calcul de la matrice d'intrinsèques (mtx) et des coefficients de distorsion (dist)
print("Calibration en cours.")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

print("\nRÉSULTATS : ")
print("Matrice intrinsèque de la caméra :\n", mtx)
print("\nCoefficients de distorsion :\n", dist)




# Extraction de la focale en mètres
# Dimensions physiques du capteur en millimètres 

# On a une ouverture f/1.78 et on utilise la caméra principale du téléphone (iPhone 15 Pro)
aperture_width = 7.3 #5.6 #9.8  # Largeur du capteur en mm
aperture_height = 9.8 #4.2 #7.6 # Hauteur du capteur en mm

# Utilisation de la matrice (mtx) et de la taille de l'image (gray_shape)
fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(
    mtx, gray_shape, aperture_width, aperture_height
)

# Conversion de la focale de millimètres vers mètres
focal_length_meters = focalLength / 1000.0

print("\nRÉSULTATS (FOCALE) : ")
print(f"Focale calculée (en mm) : {focalLength:.2f} mm")
print(f"Focale demandée (en mètres) : {focal_length_meters:.5f} m")
print(f"Champ de vision (FOV) Horizontal : {fovx:.2f}°")
print(f"Champ de vision (FOV) Vertical : {fovy:.2f}°")




# Estimation de Pose (Repère 3D)

def draw_axes(img, corners, imgpts):
    # Récupération du point d'origine (le premier coin détecté du damier)
    corner = tuple(corners[0].ravel().astype(int))
    # Dessin des 3 axes (X=Bleu, Y=Vert, Z=Rouge)
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0, 0, 255), 5)
    return img


# Définition des points 3D pour les axes (longueur = 3 cases du damier)
axis = np.float32([[3*SQUARE_SIZE, 0, 0], [0, 3*SQUARE_SIZE, 0], [0, 0, -3*SQUARE_SIZE]]).reshape(-1,3)

# Test sur une image
test_img_path = images[0]
img_pose = cv2.imread(test_img_path)
gray_pose = cv2.cvtColor(img_pose, cv2.COLOR_BGR2GRAY)

# On doit redétecter les coins spécifiquement pour cette image
ret_pose, corners_pose = cv2.findChessboardCorners(gray_pose, DAMIER, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

if ret_pose:
    corners2_pose = cv2.cornerSubPix(gray_pose, corners_pose, (11, 11), (-1, -1), criteria)
    
    # Calcul de la pose de la caméra (Vecteurs de rotation et translation)
    ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners2_pose, mtx, dist)
    
    # Projection des points 3D de l'axe vers l'image 2D
    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
    
    # Dessin des axes sur l'image
    img_pose = draw_axes(img_pose, corners2_pose, imgpts)
    
    cv2.imshow('Estimation de pose - Axes 3D', img_pose)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()




# Calcul de l'erreur moyenne de reprojection
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

reprojection_error = mean_error / len(objpoints)
print(f"\nErreur moyenne de reprojection : {reprojection_error:.4f} pixels")
print(f"Erreur RMS globale : {ret:.4f} pixels")