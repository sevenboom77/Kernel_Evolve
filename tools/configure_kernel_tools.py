from pathlib import Path
import sys,json,argparse
p=argparse.ArgumentParser();p.add_argument('--run-id',default='optimize_001');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=root/'outputs';out.mkdir(exist_ok=True)
file=out/'kernel-tools.json';file.write_text(json.dumps({'kl_python':sys.executable,'kl_root':str(root),'run_id':a.run_id},indent=2)+'\n')
print(file)
