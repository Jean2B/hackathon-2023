# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 19:43:18 2023

@author: jacqu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

# Supposons que similarity_matrix est une matrice de similarités cosinus entre les articles
# Elle peut être calculée à l'aide de la bibliothèque scikit-learn ou toute autre méthode appropriée

# Exemple de similarité cosinus entre articles (une matrice carrée)
similarity_matrix = [
    [1, 0.746, 0.739, 0.763, 0.772,	0.793, 0.807, 0.785, 0.826, 0.79, 0.754, 0.736, 0.776, 0.85, 0.766,	0.821, 0.863, 0.738, 0.814, 0.772],
    [0.746, 1, 0.907, 0.905, 0.815, 0.814, 0.845, 0.843, 0.754, 0.851, 0.874, 0.924, 0.831, 0.746, 0.858, 0.723, 0.74, 0.95, 0.741, 0.765],
    [0.739, 0.907, 1, 0.868, 0.788, 0.783, 0.823, 0.815, 0.734, 0.807, 0.866, 0.941, 0.786, 0.724, 0.808, 0.713, 0.73, 0.892, 0.707, 0.725],
    [0.763, 0.906, 0.868, 1, 0.889, 0.883, 0.81, 0.894, 0.776, 0.902, 0.905, 0.853, 0.882, 0.779, 0.904, 0.765, 0.758, 0.926, 0.754, 0.781],
    [0.772, 0.815, 0.788, 0.889, 1, 0.948, 0.776, 0.935, 0.793, 0.933, 0.85, 0.769, 0.931, 0.807, 0.938, 0.778, 0.77, 0.814, 0.76, 0.8],
    [0.793, 0.814, 0.783, 0.883, 0.948, 1, 0.783, 0.938, 0.797, 0.94, 0.843, 0.77, 0.932, 0.816, 0.915, 0.803, 0.792, 0.815, 0.787, 0.854],
    [0.807, 0.845, 0.823, 0.81, 0.776, 0.783, 1, 0.78, 0.793, 0.795, 0.814, 0.838, 0.787, 0.786, 0.789, 0.772, 0.801, 0.837, 0.783, 0.764],
    [0.785, 0.843, 0.815, 0.894, 0.935, 0.938, 0.78, 1, 0.796, 0.937, 0.855, 0.8, 0.918, 0.813, 0.935, 0.794, 0.769, 0.841, 0.763, 0.813],
    [0.826, 0.754, 0.734, 0.775, 0.793, 0.797, 0.793, 0.796, 1, 0.791, 0.77, 0.73, 0.79, 0.789, 0.787, 0.774, 0.792, 0.752, 0.803, 0.741],
    [0.79, 0.851, 0.807, 0.902, 0.933, 0.94, 0.795, 0.937, 0.791, 1, 0.873, 0.801, 0.927, 0.821, 0.923, 0.794, 0.794, 0.853, 0.788, 0.823],
    [0.754, 0.874, 0.866, 0.905, 0.85, 0.843, 0.814, 0.855, 0.77, 0.873, 1, 0.863, 0.845, 0.749, 0.85, 0.72, 0.749, 0.894, 0.73, 0.746],
    [0.736, 0.924, 0.941, 0.853, 0.769, 0.77, 0.838, 0.8, 0.73, 0.801, 0.863, 1, 0.785, 0.726, 0.806, 0.704, 0.732, 0.902, 0.72, 0.729],
    [0.776, 0.831, 0.787, 0.882, 0.931, 0.932, 0.787, 0.918, 0.79, 0.927, 0.845, 0.785, 1, 0.8, 0.924, 0.781, 0.792, 0.839, 0.772, 0.808],
    [0.85, 0.746, 0.724, 0.779, 0.807, 0.816, 0.786, 0.813, 0.789, 0.821, 0.749, 0.726, 0.8, 1, 0.783, 0.886, 0.825, 0.735, 0.809, 0.793],
    [0.766, 0.858, 0.808, 0.905, 0.938, 0.915, 0.789, 0.935, 0.787, 0.923, 0.85, 0.806, 0.924, 0.783, 1, 0.768, 0.765, 0.857, 0.759, 0.808],
    [0.821, 0.723, 0.713, 0.765, 0.778, 0.803, 0.772, 0.794, 0.774, 0.794, 0.72, 0.704, 0.781, 0.886, 0.768, 1, 0.825, 0.723, 0.816, 0.798],
    [0.863, 0.74, 0.73, 0.757, 0.77, 0.792, 0.801, 0.769, 0.792, 0.78, 0.749, 0.732, 0.792, 0.825, 0.765, 0.825, 1, 0.738, 0.826, 0.746],
    [0.738, 0.95, 0.892, 0.927, 0.814, 0.815, 0.837, 0.841, 0.752, 0.853, 0.894, 0.902, 0.839, 0.735, 0.857, 0.723, 0.738, 1, 0.755, 0.754],
    [0.814, 0.741, 0.707, 0.753, 0.76, 0.787, 0.783, 0.763, 0.803, 0.788, 0.73, 0.72, 0.772, 0.809, 0.759, 0.816, 0.826, 0.755, 1, 0.775],
    [0.772, 0.765, 0.725, 0.78, 0.8, 0.854, 0.764, 0.813, 0.741, 0.823, 0.746, 0.729, 0.808, 0.793, 0.808, 0.798, 0.746, 0.754, 0.775, 1]

]
# Convert the similarity matrix to a NumPy array
similarity_matrix = np.array(similarity_matrix)

# Compute the dissimilarity matrix
dissimilarity_matrix = 0.5 * (1 - similarity_matrix + 1 - similarity_matrix.T)

# Utilisation de MDS pour réduire la dimension à 2 pour la visualisation
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
pos = mds.fit_transform(dissimilarity_matrix)  # Utilisation de la dissimilarité plutôt que de la similarité

# Génération d'un nuage de points avec Matplotlib
plt.scatter(pos[:, 0], pos[:, 1])

# Ajout des labels pour chaque point
for i in range(len(pos)):
    plt.annotate(str(i+1), (pos[i, 0], pos[i, 1]))

# Ajout des lignes reliant les points avec des valeurs de similarité
#for i in range(len(similarity_matrix)):
#    for j in range(i + 1, len(similarity_matrix)):
#        plt.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color='gray', linestyle='--')

plt.title('Nuage de Points des Similarités Cosinus entre Articles')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()