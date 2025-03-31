import subprocess
import sys

def build_package():
    # Erst die Source Distribution (.tar.gz) erstellen
    subprocess.run([sys.executable, "-m", "build", "--sdist"], check=True)
    print("\nSource Distribution (.tar.gz) wurde erstellt.")
    
    # Dann das Wheel (.whl) erstellen
    subprocess.run([sys.executable, "-m", "build", "--wheel"], check=True)
    print("\nWheel (.whl) wurde erstellt.")
    
    print("\nBuild abgeschlossen. Die Dateien befinden sich im 'dist'-Ordner.")

if __name__ == "__main__":
    build_package()

