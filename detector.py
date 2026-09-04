import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, convolve

class SpliceBusterDetector:
    def __init__(self, window_size=32, threshold=2.0):
        """
        Inizializza il detector con i parametri da ottimizzare.
        - window_size: dimensione della finestra scorrevole per il calcolo della varianza.
        - threshold: soglia di anomalia (es. Z-score) per determinare la manomissione.
        """
        self.window_size = window_size
        self.threshold = threshold
        
        # Filtro passa-alto standard (derivato da SRM/Laplaciano) per estrarre il rumore residuo
        self.high_pass_filter = np.array([
            [-1,  2, -1],
            [ 2, -4,  2],
            [-1,  2, -1]
        ]) / 4.0

    def get_noise_residual(self, image_array):
        """Applica il filtro passa-alto per isolare il rumore."""
        # Convertiamo in float32 per evitare overflow durante i calcoli
        img_f = image_array.astype(np.float32)
        residual = convolve(img_f, self.high_pass_filter)
        return residual

    def get_local_variance(self, residual):
        """Calcola la varianza locale in modo efficiente usando una sliding window."""
        sq_residual = residual ** 2
        
        # E[X^2] - (E[X])^2
        mean_sq = uniform_filter(sq_residual, size=self.window_size)
        sq_mean = uniform_filter(residual, size=self.window_size) ** 2
        
        variance = mean_sq - sq_mean
        # Evitiamo varianze negative causate da approssimazioni in virgola mobile
        return np.clip(variance, a_min=0, a_max=None)

    def predict(self, image_path):
        """
        Processa un'immagine e restituisce la mappa di anomalia continua e la maschera binaria.
        """
        # 1. Caricamento in scala di grigi (il rumore strutturale si analizza bene in luma)
        with Image.open(image_path) as img:
            img_gray = np.array(img.convert('L'))
            
        # 2. Estrazione del rumore residuo
        residual = self.get_noise_residual(img_gray)
        
        # 3. Calcolo della varianza spaziale (applica la window_size)
        variance_map = self.get_local_variance(residual)
        
        # 4. Normalizzazione (Z-score dell'immagine)
        # Una manomissione è un'anomalia statistica rispetto al resto dell'immagine
        mean_v = np.mean(variance_map)
        std_v = np.std(variance_map) + 1e-8
        z_map = (variance_map - mean_v) / std_v
        
        # Prendiamo il valore assoluto perché la manomissione può generare
        # rumore più alto (es. splicing non compresso) o più basso (es. area sfuocata)
        anomaly_map = np.abs(z_map)
        
        # 5. Applica la soglia (applica il threshold)
        binary_mask = (anomaly_map > self.threshold).astype(np.uint8)
        
        return anomaly_map, binary_mask

    def update_parameters(self, new_window_size=None, new_threshold=None):
        """Permette di aggiornare i parametri durante la fase di ottimizzazione."""
        if new_window_size is not None:
            self.window_size = int(new_window_size)
        if new_threshold is not None:
            self.threshold = float(new_threshold)