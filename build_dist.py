import subprocess
import shutil
import sys
import os

def clean_build_artifacts():
    """Temporäre Build-Verzeichnisse entfernen"""
    artifacts = ["WebLlama.egg-info", "build"]
    
    for artifact in artifacts:
        if os.path.exists(artifact):
            try:
                if os.path.isdir(artifact):
                    shutil.rmtree(artifact)
                else:
                    os.remove(artifact)
            except Exception as e:
                print(f"Fehler beim Entfernen von {artifact}: {e}")

def build_package():
    # Zuerst temporäre Build-Verzeichnisse entfernen
    print("Entferne alte Build-Verzeichnisse...", end="\r")
    clean_build_artifacts()
    
    # Erst die Source Distribution (.tar.gz) erstellen
    subprocess.run([sys.executable, "-m", "build", "--sdist"], check=True)
    print("\nSource Distribution (.tar.gz) wurde erstellt.")
    
    # Dann das Wheel (.whl) erstellen
    subprocess.run([sys.executable, "-m", "build", "--wheel"], check=True)
    print("\nWheel (.whl) wurde erstellt.")
    
    # Aufräumen nach dem Build
    print("\nEntferne temporäre Build-Verzeichnisse...", end="\r")
    clean_build_artifacts()
    
    print("\nBuild abgeschlossen. Die Dateien befinden sich im 'dist'-Ordner.")

if __name__ == "__main__":
    build_package()
