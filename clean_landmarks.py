import os
import json
import re
from collections import Counter

def convertir_label_dentaire(label_actuel):
    """
    Associe les labels continus L-1 à L-14 vers la nomenclature LL/LR.
    """
    # Dictionnaire de correspondance (Mapping)
    mapping = {
        "L-1": "LL7", "L-2": "LL6", "L-3": "LL5", "L-4": "LL4", 
        "L-5": "LL3", "L-6": "LL2", "L-7": "LL1",
        "L-8": "LR1", "L-9": "LR2", "L-10": "LR3", "L-11": "LR4", 
        "L-12": "LR5", "L-13": "LR6", "L-14": "LR7"
    }
    
    # On extrait la fin du label (ex: "H2_T1_L-10" -> "L-10")
    import re
    match = re.search(r'L-\d+$', label_actuel)
    
    if match:
        cle_l = match.group() # Donne "L-10"
        if cle_l in mapping:
            return mapping[cle_l] # Retourne "LR3"
            
    return label_actuel # Retourne le label d'origine si non trouvé

def normaliser_dossier_json(chemin_dossier):
    # Vérifie si le dossier existe
    if not os.path.exists(chemin_dossier):
        print(f"Erreur : Le dossier '{chemin_dossier}' n'existe pas.")
        return

    # Parcourt tous les fichiers du dossier
    for nom_fichier in os.listdir(chemin_dossier):
        if nom_fichier.endswith('.json'):
            chemin_fichier = os.path.join(chemin_dossier, nom_fichier)
            
            # 1. Lecture du fichier JSON
            try:
                with open(chemin_fichier, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Impossible de lire le fichier {nom_fichier} : {e}")
                continue

            # 2. Modification des labels dans la structure 3D Slicer Markups
            modifie = False
            if "markups" in data:
                for markup in data["markups"]:
                    if "controlPoints" in markup:
                        for cp in markup["controlPoints"]:
                            if "label" in cp:
                                ancien_label = cp["label"]
                                nouveau_label = convertir_label_dentaire(ancien_label)
                                if "LL" in nouveau_label:
                                    nouveau_label=nouveau_label.replace("LL","LR")
                                elif "LR" in nouveau_label:
                                    nouveau_label=nouveau_label.replace("LR","LL")
                                
                                if ancien_label=="F-1MG":
                                    print(nom_fichier)
                                if ancien_label != nouveau_label:
                                    cp["label"] = nouveau_label
                                    modifie = True

            # 3. Sauvegarde si des modifications ont eu lieu
            if modifie:
                try:
                    with open(chemin_fichier, 'w', encoding='utf-8') as f:
                        # indent=4 permet de garder le JSON lisible et bien formaté
                        json.dump(data, f, indent=4, ensure_ascii=False)
                    print(f"✓ Fichier normalisé avec succès : {nom_fichier}")
                except Exception as e:
                    print(f"Erreur lors de la sauvegarde de {nom_fichier} : {e}")
            else:
                print(f"• Rien à changer pour : {nom_fichier}")

def compter_presence_labels(chemin_dossier):
    # Vérifie si le dossier existe
    if not os.path.exists(chemin_dossier):
        print(f"Erreur : Le dossier '{chemin_dossier}' n'existe pas.")
        return

    compteur_global = Counter()
    total_fichiers = 0

    # Parcourt le dossier pour compter
    for nom_fichier in os.listdir(chemin_dossier):
        if nom_fichier.endswith('.json'):
            chemin_fichier = os.path.join(chemin_dossier, nom_fichier)
            total_fichiers += 1
            
            try:
                with open(chemin_fichier, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extraction des labels
                if "markups" in data:
                    for markup in data["markups"]:
                        if "controlPoints" in markup:
                            for cp in markup["controlPoints"]:
                                if "label" in cp and cp["position"]!="":
                                    # print(cp["position"])
                                    if cp['label'] == "LL7MG":
                                        print(nom_fichier)
                                    compteur_global[cp["label"]] += 1
                                    
            except Exception as e:
                print(f"Erreur de lecture sur le fichier {nom_fichier} : {e}")

    # --- AFFICHAGE DES RÉSULTATS ---
    print("\n" + "="*50)
    print(f"RÉSUMÉ DES LANDMARKS ({total_fichiers} fichiers analysés)")
    print("="*50)
    print(f"'Nom du Label' | 'Nombre d\'apparitions'")
    print("-"*50)
    
    # Tri des résultats par nom de label (ou par fréquence en changeant par .most_common())
    for label, compte in sorted(compteur_global.items()):
        print(f"{label} | {compte}")
        
    print("="*50)
    print(f"Total de points détectés : {sum(compteur_global.values())}\n")

if __name__ == "__main__":
    dossier_cible = "/home/luciacev/Desktop/training ios files/mucogingival/landmarks" 
    
    
    normaliser_dossier_json(dossier_cible)
    compter_presence_labels(dossier_cible)