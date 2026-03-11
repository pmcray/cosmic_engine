import json
import os

def create_notebook(filename, cells):
    nb = {
        "cells": [],
        "metadata": {
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    for cell_type, source in cells:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": [line + "\n" for line in source.split('\n')[:-1]] + [source.split('\n')[-1]] if source else []
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb["cells"].append(cell)
        
    with open(filename, 'w') as f:
        json.dump(nb, f, indent=2)

os.makedirs("notebooks", exist_ok=True)

# 1. Infinite Director Colab Notebook
director_cells = [
    ("markdown", "# Infinite Stargate Sequence on Google Colab\n\nThis notebook sets up the environment and runs the `infinite_director.py` pipeline to generate cinematic stargate sequences."),
    ("code", "!pip install taichi opencv-python numpy scipy scikit-image\n!apt-get install -y ffmpeg"),
    ("markdown", "## Setup Repositories\nAssuming the repositories are either cloned here or mounted via Google Drive.\nFor Colab, you might want to zip and upload your local `cosmic_engine`, `weorold`, and `worldmaker` folders, or mount your drive."),
    ("code", "from google.colab import drive\ndrive.mount('/content/drive')\n\n# NOTE: Adjust these paths to where your repos are stored in Drive!\nimport sys\nsys.path.append('/content/drive/MyDrive/worldmaker')\nsys.path.append('/content/drive/MyDrive/weorold')\nsys.path.append('/content/drive/MyDrive/cosmic_engine')"),
    ("markdown", "## Run the Pipeline"),
    ("code", "import os\n# Move to engine directory (adjust path as needed)\nos.chdir('/content/drive/MyDrive/cosmic_engine')\n\n!python infinite_director.py"),
    ("markdown", "## Display the Result"),
    ("code", "from IPython.display import HTML\nfrom base64 import b64encode\n\nvid_path = 'director_output/final_sequence.mp4'\nmp4 = open(vid_path,'rb').read()\ndata_url = \"data:video/mp4;base64,\" + b64encode(mp4).decode()\nHTML(f\"\"\"\n<video width=800 controls>\n      <source src=\"{data_url}\" type=\"video/mp4\">\n</video>\n\"\"\")")
]

create_notebook("notebooks/Colab_Infinite_Director.ipynb", director_cells)

# 2. Interactive Exotic Physics
exotic_cells = [
    ("markdown", "# Interactive Exotic Physics Testing\n\nTest individual geodesic black holes or megastructures."),
    ("code", "!pip install taichi opencv-python numpy diffusers transformers accelerate\n!apt-get install -y ffmpeg"),
    ("code", "from google.colab import drive\ndrive.mount('/content/drive')\nimport os\nimport sys\nos.chdir('/content/drive/MyDrive/cosmic_engine')"),
    ("code", "import taichi as ti\nti.init(arch=ti.gpu)\n\nfrom encounters.exotic_physics import ExoticPhysicsEncounter\n\n# Generate a Blackhole fly-by\nenc = ExoticPhysicsEncounter('blackhole', res=512)\nenc.render_to_mp4('blackhole_test.mp4', total_frames=60)"),
    ("code", "from IPython.display import HTML\nfrom base64 import b64encode\n\nmp4 = open('blackhole_test.mp4','rb').read()\ndata_url = \"data:video/mp4;base64,\" + b64encode(mp4).decode()\nHTML(f\"\"\"\n<video width=800 controls>\n      <source src=\"{data_url}\" type=\"video/mp4\">\n</video>\n\"\"\")"),
    ("markdown", "## Megastructures Testing"),
    ("code", "from encounters.megastructure import MegastructureEncounter\n\n# Generate an Alien Megastructure (L-System + Stable Diffusion)\nmega_enc = MegastructureEncounter('alien_structure', res=512)\nmega_enc.render_to_mp4('megastructure_test.mp4', test_mode=True)\n\nmp4_mega = open('megastructure_test.mp4','rb').read()\ndata_url_mega = \"data:video/mp4;base64,\" + b64encode(mp4_mega).decode()\nHTML(f\"\"\"\n<video width=800 controls>\n      <source src=\"{data_url_mega}\" type=\"video/mp4\">\n</video>\n\"\"\")")
]

create_notebook("notebooks/Colab_Interactive_Exotic_Physics.ipynb", exotic_cells)

print("Notebooks created in notebooks/")
