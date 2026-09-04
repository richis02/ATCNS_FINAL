import numpy as np
from PIL import Image
import detector # Importa il file detector.py che abbiamo creato prima

def calculate_confusion_matrix(pred_mask, true_mask):
    """Calcola i pixel True Positive, False Positive e False Negative."""
    p = (pred_mask > 0)
    t = (true_mask > 0)
    
    tp = np.sum(p & t)
    fp = np.sum(p & ~t)
    fn = np.sum(~p & t)
    
    return tp, fp, fn

def grid_search_calibration(val_set, detector_instance, window_sizes, thresholds):
    """
    Esegue una Grid Search sul Validation Set per trovare la combinazione
    (window_size, threshold) che massimizza l'F1-score globale.
    """
    print(f"Inizio calibrazione su {len(val_set)} immagini...")
    
    # Inizializziamo i contatori globali per ogni combinazione (w, t)
    # L'approccio globale (micro-average) è molto più stabile se ci sono maschere piccole
    results = {(w, t): {'tp': 0, 'fp': 0, 'fn': 0} for w in window_sizes for t in thresholds}
    
    for i, item in enumerate(val_set):
        img_path = item['image']
        mask_path = item['mask']
        
        # Carica la maschera Ground Truth e l'immagine
        with Image.open(mask_path) as m:
            true_mask = np.array(m.convert('L')) > 127
            
        with Image.open(img_path) as img:
            img_gray = np.array(img.convert('L'))
            
        for w in window_sizes:
            # 1. Aggiorna la finestra nel detector
            detector_instance.update_parameters(new_window_size=w)
            
            # 2. Calcola la mappa delle anomalie UNA SOLA VOLTA per questa finestra
            residual = detector_instance.get_noise_residual(img_gray)
            variance_map = detector_instance.get_local_variance(residual)
            
            mean_v = np.mean(variance_map)
            std_v = np.std(variance_map) + 1e-8
            anomaly_map = np.abs((variance_map - mean_v) / std_v)
            
            # 3. Testa tutte le soglie istantaneamente su questa mappa
            for t in thresholds:
                pred_mask = anomaly_map > t
                tp, fp, fn = calculate_confusion_matrix(pred_mask, true_mask)
                
                results[(w, t)]['tp'] += tp
                results[(w, t)]['fp'] += fp
                results[(w, t)]['fn'] += fn
                
        if (i + 1) % 25 == 0 or (i + 1) == len(val_set):
            print(f"Elaborate {i + 1}/{len(val_set)} immagini...")

    # Ora calcoliamo l'F1 finale per ogni combinazione
    best_f1 = -1
    best_params = {'window_size': None, 'threshold': None}
    
    for w in window_sizes:
        for t in thresholds:
            tp = results[(w, t)]['tp']
            fp = results[(w, t)]['fp']
            fn = results[(w, t)]['fn']
            
            # Formula F1 = 2*TP / (2*TP + FP + FN)
            if (tp + fp + fn) > 0:
                f1 = (2.0 * tp) / (2.0 * tp + fp + fn)
            else:
                f1 = 0.0
                
            if f1 > best_f1:
                best_f1 = f1
                best_params = {'window_size': w, 'threshold': t}

    print("-" * 40)
    print(f"CALIBRAZIONE COMPLETATA!")
    print(f"Migliori parametri trovati: Finestra = {best_params['window_size']}, Soglia = {best_params['threshold']}")
    print(f"Miglior F1-Score: {best_f1:.4f}")
    
    return best_params, best_f1