"""Run reproducible data foundation from the original unmodified generator."""
import subprocess
import sys
from pathlib import Path
root=Path(__file__).resolve().parent
for name in ['generate_data.py','part_a.py','validate_part_a.py','analyze_part_a.py']:
    subprocess.run([sys.executable,str(root/name)],cwd=root,check=True)
