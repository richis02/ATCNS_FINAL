import os
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

def is_valid_image(file_path):
    """Verifica che il file immagine/maschera si possa aprire correttamente."""
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verifica l'integrità senza caricare tutti i pixel in RAM
        return True
    except Exception:
        return False

def load_and_filter_dataset(dataset_root, img_folder_name= "tp", mask_folder_name="groundtruth", mask_suffix="_gt"):
    """
    Abbina le immagini alle maschere e scarta quelle non valide.
    Adattato per la struttura tipica di CASIA 2.0.
    """
    root_path = Path(dataset_root)
    # Supponiamo che le immagini manomesse siano nella cartella 'Tp'
    tp_folder = root_path / img_folder_name
    mask_folder = root_path / mask_folder_name
    
    valid_pairs = []
    discarded_count = 0
    
    if not tp_folder.exists() or not mask_folder.exists():
        raise FileNotFoundError("Cartelle delle immagini (Tp) o delle maschere non trovate. Controlla il path.")
    
    # Scorre tutte le immagini nella cartella Tp
    for img_path in tp_folder.glob("*.*"):
        if img_path.suffix.lower() not in ['.jpg', '.png', '.tif']:
            continue
            
        # Costruisce il nome atteso della maschera (es. Tp_D_CNN_001.jpg -> Tp_D_CNN_001_gt.png)
        expected_mask_name = f"{img_path.stem}{mask_suffix}.png"
        mask_path = mask_folder / expected_mask_name
        
        # Verifica 1: La maschera esiste?
        # Verifica 2: L'immagine e la maschera si aprono senza errori?
        if mask_path.exists() and is_valid_image(img_path) and is_valid_image(mask_path):
            valid_pairs.append({
                "image": str(img_path),
                "mask": str(mask_path)
            })
        else:
            discarded_count += 1
            
    print(f"Dataset caricato: {len(valid_pairs)} coppie valide trovate.")
    print(f"Immagini scartate (maschera assente o file corrotto): {discarded_count}")
    
    return valid_pairs

def split_dataset(pairs, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Divide la lista di coppie valide in Train, Validation e Test set.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-5, "Le percentuali di split devono sommare a 1."
    
    # Primo split: separiamo il Train dal resto (Val + Test)
    train_pairs, temp_pairs = train_test_split(
        pairs, 
        train_size=train_size, 
        random_state=random_state
    )
    
    # Calcoliamo la proporzione del Validation rispetto al rimanente
    val_ratio = val_size / (val_size + test_size)
    
    # Secondo split: separiamo Val e Test
    val_pairs, test_pairs = train_test_split(
        temp_pairs, 
        train_size=val_ratio, 
        random_state=random_state
    )
    
    print(f"Split completato: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
    
    return train_pairs, val_pairs, test_pairs