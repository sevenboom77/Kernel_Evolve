"""Time the GPU-qualified fused Welford repair with the original paired protocol."""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
DIAGNOSTIC_SHA='e014aaadc10f19df106f9e60e3aba18270f12cfa487febe25b178da80da3ba76'

def reference(path):
 path=Path(path).resolve();return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def bound(ref):
 if reference(ref['path'])!=ref:raise ValueError('numerical qualification source or evidence changed')
 return json.loads(Path(ref['path']).read_bytes())

def load_diagnostic():
 p=HERE/'diagnose_pretrained_gpt2_welford.py'
 if reference(p)['sha256']!=DIAGNOSTIC_SHA:raise ValueError('GPU-qualified diagnostic source changed')
 s=importlib.util.spec_from_file_location('private_Welford_performance_diagnostic',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def verify_numerical_qualification(path,sha256):
 execution_ref={'path':str(Path(path).resolve()),'sha256':sha256};value=bound(execution_ref)
 if (value.get('status')!='ALL_ORIGINAL_FOUR_NUMERICAL_CASES_CLOSED' or value.get('exit_code')!=0 or value.get('all_four_full_model_and_48_local_checks_correct') is not True or value.get('owned_RPC_formally_closed') is not True or value.get('foreign_processes_signaled')!=0):raise ValueError('four original numerical cases and owned closure required')
 freeze=bound(value['source_freeze'])
 for ref in freeze['sources']:
  if reference(ref['path'])!=ref:raise ValueError('original numerical source changed')
 qualification=bound(value['qualification']);primary=qualification['primary_closure']['completion'];closure=load_diagnostic().verify_closed_primary(primary['path'],primary['sha256'])
 labels=[];local=0
 for item in value['reports']:
  report=bound(item['report'])
  if (report['status']!='DIAGNOSED_CLEAN' or not report['actual_fused_Welford_CUDA_invoked'] or not report['all_pretrained_weights_preserved'] or report['parameter_count']!=124439808 or report['parameter_sha256']!=value['parameter_sha256']):raise ValueError('original GPU numerical reports required')
  if report['protocol']['kernel_tolerance']!={'rtol':.002,'atol':.01} or report['protocol']['logit_tolerance']!={'rtol':.005,'atol':.05}:raise ValueError('original tolerances changed')
  for label,row in report['measurements'].items():
   labels.append(label);checks=row['fused_welford_checks'];local+=len(checks)
   if len(checks)!=12 or any(x['status']!='AVAILABLE' for x in checks) or any(x['status']!='AVAILABLE' for x in row['backend_qualification'].values()):raise ValueError('all local and full numerical checks must pass')
 if sorted(labels)!=['B1_S128','B1_S33','B1_S512','B2_S1024'] or local!=48:raise ValueError('four unique original numerical cases required')
 return {'execution':execution_ref,'reports':value['reports'],'all_four_correct':True,'local_checks':48,'primary_closure':closure,'original_numerical_sources_unchanged':True}

def performance_protocol(base):
 protocol=copy.deepcopy(base.make_protocol());protocol.update(name='pretrained-gpt2-fused-Welford-full-forward-v1',optimization='GPU-qualified fused CUDA Welford residual plus ln_2 in all twelve pretrained blocks',measurement_requires_complete_numerical_qualification=True)
 return protocol

def label_actual_variant(report):
 if 'actual_K02_Triton_executed' in report:
  actual=report.pop('actual_K02_Triton_executed');report['actual_fused_Welford_CUDA_executed']=actual;report['actual_fused_Welford_CUDA_invoked']=actual
 for row in (report.get('measurements') or {}).values():
  if 'original_k02_checks' in row:row['fused_Welford_checks']=row.pop('original_k02_checks')
 if report.get('status')=='MEASURED_CLEAN':report['performance_measured']=True


def adapt_private_base(base,qualification=None):
 protocol=performance_protocol(base);original_gpu=base.run_gpu;original_atomic=base.atomic_json
 base=load_diagnostic().adapt_private_base(base,qualification['primary_closure'] if qualification else None)
 source_package,make_plan=base.source_package,base.make_plan
 base.make_protocol=lambda:copy.deepcopy(protocol)
 def package(*args):
  value,_=source_package(*args);value['application_sources'][str(Path(__file__).resolve().relative_to(base.ROOT))]=reference(__file__)['sha256']
  return value,hashlib.sha256(base.canonical(value)).hexdigest()
 def plan(*args):
  report=make_plan(*args);report.update(kind='source-bound-pretrained-gpt2-fused-Welford-performance',numerical_qualification=qualification,performance_measured=False)
  return report
 def atomic(path,report):
  label_actual_variant(report);return original_atomic(path,report)
 base.source_package,base.make_plan,base.atomic_json,base.run_gpu=package,plan,atomic,original_gpu
 return base

def main(argv=None):
 argv=list(sys.argv[1:] if argv is None else argv);p=argparse.ArgumentParser(add_help=False);p.add_argument('--numerical-qualification',type=Path);p.add_argument('--numerical-qualification-sha256');args,remaining=p.parse_known_args(argv)
 qualification=None
 if '--execute' in remaining:
  if args.numerical_qualification is None or args.numerical_qualification_sha256 is None:raise ValueError('four original numerical case qualification required before GPU timing')
  qualification=verify_numerical_qualification(args.numerical_qualification,args.numerical_qualification_sha256)
 return adapt_private_base(load_diagnostic().load_frozen_base(),qualification).main(remaining)
if __name__=='__main__':raise SystemExit(main())
