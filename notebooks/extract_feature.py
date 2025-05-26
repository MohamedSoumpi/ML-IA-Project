import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage import center_of_mass, rotate as ndrotate
import pandas as pd
import cv2
import os
from glob import glob
from skimage.feature import graycomatrix, graycoprops


def clean_mask(mask):
    """Nettoie le masque en gardant la plus grande forme blanche"""
    _, labels = cv2.connectedComponents(mask.astype(np.uint8))
    unique, counts = np.unique(labels, return_counts=True)
    counts[0] = 0  # ignorer le fond
    main_label = unique[np.argmax(counts)]
    cleaned = (labels == main_label).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(cleaned)
    return cleaned[y:y+h, x:x+w], (x, y, w, h)

def get_best_inscribed_circle(mask):
    """Retourne l’aire du plus grand cercle inscrit dans le corps"""
    mask_clean, _ = clean_mask(mask)
    dist_transform = cv2.distanceTransform(mask_clean, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
    radius = dist_transform[y, x]
    area = np.pi * radius ** 2
    return area

#def rotate_image_center(img, angle, center):
 #   """Tourne une image autour d’un point (xc, yc)"""
  #  angle = float(angle)  # ✅ assure que l'angle est un float simple
   # (h, w) = img.shape[:2]
   # M = cv2.getRotationMatrix2D(center, angle, 1.0)
    #rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    #return rotated


def symmetry_loss(angle, img, center):
    """Fonction de perte pour comparer une image à son miroir"""
    angle = float(angle)
    rotated = ndrotate(img, angle, reshape=False, order=1, mode='constant', cval=0)

    # Utilise le centre de l’image pour couper
    xc = rotated.shape[1] // 2  # moité de la largeur
    left = rotated[:, :xc]
    right = np.fliplr(rotated[:, xc:])

    # On garde la même taille des deux côtés
    min_width = min(left.shape[1], right.shape[1])
    left = left[:, :min_width]
    right = right[:, :min_width]

    # Calcul de l'erreur moyenne
    diff = (left - right) ** 2
    return diff.mean()


def get_symmetry_error(img, mask):
    """Retourne l’erreur de symétrie minimale"""
    mask_clean, (x, y, w, h) = clean_mask(mask)
    img_crop = img[y:y+h, x:x+w]
    center = center_of_mass(mask_clean)
    gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
    res = minimize(symmetry_loss, x0=[0], args=(gray, center), method='Powell')
    return res.fun

def extract_features(img_path, mask_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask_bool = mask > 0

    ratio = np.sum(mask_bool) / mask.size
    pixels = img_rgb[mask_bool]

    features = {
        'ID': os.path.basename(img_path).split('.')[0],
        'ratio': ratio,
        'min_R': pixels[:,0].min(), 'min_G': pixels[:,1].min(), 'min_B': pixels[:,2].min(),
        'max_R': pixels[:,0].max(), 'max_G': pixels[:,1].max(), 'max_B': pixels[:,2].max(),
        'mean_R': pixels[:,0].mean(), 'mean_G': pixels[:,1].mean(), 'mean_B': pixels[:,2].mean(),
        'median_R': np.median(pixels[:,0]), 'median_G': np.median(pixels[:,1]), 'median_B': np.median(pixels[:,2]),
        'std_R': pixels[:,0].std(), 'std_G': pixels[:,1].std(), 'std_B': pixels[:,2].std(),
    }

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked_gray = np.where(mask_bool, gray_img, 0).astype(np.uint8)
    glcm = graycomatrix(masked_gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    features['texture_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    features['texture_correlation'] = graycoprops(glcm, 'correlation')[0, 0]

    print(f"✔️  Test image traitée : {os.path.basename(img_path)}")
    return features
