import numpy as np
import os

def patchbin_decoder(filepath: str) -> None:
    """
    Lit un fichier binaire patch avec header (nombre de colonnes en uint8 + données uint8)
    et réécrit son contenu sous forme d'un fichier texte (ASCII) au même chemin.
    """
    with open(filepath, "rb") as f:
        num_columns = np.frombuffer(f.read(1), dtype=np.uint8)[0] # Read header (1 byte)
        data = np.frombuffer(f.read(), dtype=np.uint8) # Read remaining data

    data = data.reshape((-1, num_columns))

    # Output path
    base, _ = os.path.splitext(filepath)
    output_txt = f"{base}_decoded.txt"

    # Save as ASCII (int)
    np.savetxt(output_txt, data, fmt="%d", delimiter="\t")
    print(f"Fichier décompressé sauvegardé : {output_txt}")
    