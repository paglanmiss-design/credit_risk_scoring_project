import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

print("===== ÉTAPE 3 : PRÉ-TRAITEMENT AVANCÉ DES DONNÉES =====")

# Charger données nettoyées (chemin relatif au workspace)
data_path = os.path.join('data', 'clean_data.csv')
df = pd.read_csv(data_path)

# Étape 1 : Séparer variables explicatives (X) et variable cible à prédire (y)
# X = toutes les colonnes sauf `default` ; y = uniquement la colonne `default`
X = df.drop("default", axis=1)
y = df["default"]

# Étape 2 : Split train / test (80% / 20%)
# random_state=42 fixe l’aléatoire pour avoir des résultats reproductibles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Taille jeu d'entraînement initial : {X_train.shape[0]} clients")
print(f"Taille jeu de test : {X_test.shape[0]} clients")

# Étape 3 : Standardisation des variables numériques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Étape 4 : Équilibrer le jeu d’entraînement avec SMOTE (résolution déséquilibre classes)
# Adjust k_neighbors to the size of the minority class to avoid errors on very small datasets
n_minority = y_train.value_counts().min()
if n_minority < 2:
	print("\nPas assez d'échantillons minoritaires pour SMOTE; on saute SMOTE.")
	X_train_equilibre, y_train_equilibre = X_train_scaled, y_train
else:
	k_neighbors = min(5, n_minority - 1)
	smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
	X_train_equilibre, y_train_equilibre = smote.fit_resample(X_train_scaled, y_train)

print(f"\nTaille jeu d'entraînement après équilibrage SMOTE : {X_train_equilibre.shape[0]} clients")
print("Répartition classes après SMOTE :")
print(pd.Series(y_train_equilibre).value_counts())

# Étape 5 : Sauvegarder les données traitées sous format numpy pour la modélisation
out_dir = 'data'
os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, "X_train.npy"), X_train_equilibre)
np.save(os.path.join(out_dir, "X_test.npy"), X_test_scaled)
np.save(os.path.join(out_dir, "y_train.npy"), y_train_equilibre)
np.save(os.path.join(out_dir, "y_test.npy"), y_test)

print("\n✅ Données pré-traitées et équilibrées sauvegardées dans le dossier data")