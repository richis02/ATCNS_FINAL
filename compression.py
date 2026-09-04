import io
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def compress_jpeg(image_path, output_path, quality=75):
    """Applica una compressione JPEG standard."""
    with Image.open(image_path) as img:
        # Convertiamo in RGB nel caso ci siano immagini in scala di grigi o con canale alfa
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(output_path, "JPEG", quality=quality)
    return output_path

def compress_osn(image_path, output_path, max_size=256, quality=75):
    """
    Simula la pipeline di un Online Social Network (OSN):
    1. Downsample con Lanczos se il lato lungo supera max_size.
    2. Compressione JPEG.
    3. Upsample con Lanczos alla risoluzione originale.
    """
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        original_size = img.size
        
        # 1. Downsample
        if max(original_size) > max_size:
            ratio = max_size / float(max(original_size))
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        else:
            img_resized = img
            
        # 2. Compressione JPEG in memoria (simulazione del salvataggio sui server OSN)
        buffer = io.BytesIO()
        img_resized.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        
        # 3. Upsample alla risoluzione originale
        img_osn = Image.open(buffer)
        img_final = img_osn.resize(original_size, Image.Resampling.LANCZOS)
        
        # Salviamo l'immagine risultante (usiamo quality=100 per non aggiungere artefatti extra 
        # oltre a quelli già generati dalla simulazione OSN)
        img_final.save(output_path, "JPEG", quality=100)
        
    return output_path

def compress_jpeg_ai(image_path, output_path, compressai_model, device="cuda"):
    """
    Applica la compressione neurale tramite CompressAI (es. cheng2020_anchor)
    gestendo il padding automatico per immagini non divisibili per 64.
    """
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # 1. Convertiamo in tensore
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        
        # 2. Calcoliamo il padding necessario per rendere H e W multipli di 64
        h, w = x.size(2), x.size(3)
        p = 64  # Valore di stride massimo del modello cheng2020
        new_h = (h + p - 1) // p * p
        new_w = (w + p - 1) // p * p
        
        pad_h = new_h - h
        pad_w = new_w - w
        
        # F.pad format: (sinistra, destra, alto, basso)
        x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        
        # 3. Inferenza del modello (compressione ed espansione)
        with torch.no_grad():
            out_net = compressai_model(x_padded)
            
        # 4. Recuperiamo l'immagine ricostruita
        out_tensor = out_net['x_hat'].squeeze(0).cpu()
        out_tensor = torch.clamp(out_tensor, 0, 1)
        
        # 5. Rimuoviamo il padding ritagliando le dimensioni originali (h, w)
        out_tensor = out_tensor[:, :h, :w]
        
        # 6. Riconvertiamo in immagine PIL e salviamo
        img_recon = transforms.ToPILImage()(out_tensor)
        img_recon.save(output_path, "PNG")
        
    return output_path