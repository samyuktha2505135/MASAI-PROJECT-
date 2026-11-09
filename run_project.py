"""Reproduce the entire Masai project in the required A -> B -> C -> D order."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for script in ('generate_data.py','part_a.py','validate_part_a.py','analyze_part_a.py','part_b.py','part_c.py','part_d.py','integration_tests.py'):
    print(f'\n=== Running {script} ===',flush=True)
    subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT,check=True)
print('\nALL PROJECT STAGES COMPLETED',flush=True)
