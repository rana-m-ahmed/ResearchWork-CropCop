from __future__ import annotations

import json
from pathlib import Path


def md(text: str): return {"cell_type":"markdown","metadata":{},"source":text.splitlines(keepends=True)}
def code(text: str): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":text.splitlines(keepends=True)}


def build_notebook():
    cells=[
        md("""# CropCop Track B — Historical Comparison Index Builder\n\n**Infrastructure-only notebook; not a claim-producing evaluation.** Attach exactly one Track-B `core` package and one `historical_source` package covering the complete 117,546-image V4 audited universe. Use Kaggle **T4x2**; the builder uses `cuda:0` and four CPU workers.\n\nThe output is the immutable `historical_compare` Kaggle dataset consumed by the final Track-B notebook. It contains SHA/pHash/dHash identity, frozen DINO audit features, and packed ORB representations, not classifier predictions."""),
        md("""## Environment\n\nConfigure Kaggle Dependency Manager with `journal_extension/track_b_r07/requirements-trackb.lock.txt` before **Save & Run All**. The output must fit `/kaggle/working` and should be published as a private immutable Kaggle dataset after this builder returns PASS."""),
        code("""from pathlib import Path\nimport json, shutil, subprocess, sys\nINPUT=Path('/kaggle/input')\nOUTPUT=Path('/kaggle/working/cropcop_hist_compare_v1')\ncore=[]; hist=[]\nfor p in INPUT.glob('**/TRACKB_INPUT_MANIFEST.json'):\n    o=json.loads(p.read_text());\n    if o.get('role')=='core': core.append((p,o))\nfor p in INPUT.glob('**/TRACKB_HIST_SOURCE_MANIFEST.json'):\n    o=json.loads(p.read_text());\n    if o.get('role')=='historical_source': hist.append((p,o))\nif len(core)!=1 or len(hist)!=1: raise RuntimeError(f'Need exactly one core and one historical_source package; got core={len(core)}, historical_source={len(hist)}')\ncore_p,core_o=core[0]; hist_p,hist_o=hist[0]\nrepo=(core_p.parent/core_o['repository_root']).resolve()\nrunner=repo/'journal_extension/scripts/build_trackb_historical_compare.py'\nif not runner.is_file(): raise RuntimeError(f'Builder missing: {runner}')\ndef f(bundle_p,bundle_o,key): return (bundle_p.parent/bundle_o['files'][key]['path']).resolve()\nsource_manifest=f(hist_p,hist_o,'source_manifest')\nimage_root=(hist_p.parent/hist_o['image_root']).resolve()\ndino=f(core_p,core_o,'dino_checkpoint')\nfactory=f(core_p,core_o,'dino_factory_manifest')\nexecution_lock=f(core_p,core_o,'execution_lock')\ncode_attestation=f(core_p,core_o,'code_attestation')\nfactory_root=(core_p.parent/core_o['dino_factory_source_root']).resolve()\nprint('Free working GB:', round(shutil.disk_usage('/kaggle/working').free/1024**3,2))\nsubprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],check=True)\n"""),
        code("""if OUTPUT.exists() and any(OUTPUT.iterdir()): raise RuntimeError(f'Output must be empty: {OUTPUT}')\ncmd=[sys.executable,str(runner),'--source-manifest',str(source_manifest),'--image-root',str(image_root),'--id-column',hist_o.get('id_column','hist_id'),'--path-column',hist_o.get('path_column','relative_path'),'--dino-checkpoint',str(dino),'--factory-manifest',str(factory),'--factory-source-root',str(factory_root),'--repo-root',str(repo),'--execution-lock',str(execution_lock),'--code-attestation',str(code_attestation),'--output-dir',str(OUTPUT),'--device','cuda:0','--workers','4','--orb-chunk-size','512','--dino-batch-size','64']\nprint('Launching historical index builder')\nsubprocess.run(cmd,cwd=repo,check=True)\n"""),
        code("""manifest=json.loads((OUTPUT/'TRACKB_INPUT_MANIFEST.json').read_text())\ncert=json.loads((OUTPUT/'BUILD_CERTIFICATE.json').read_text())\nif manifest.get('role')!='historical_compare' or manifest.get('image_count')!=117546 or cert.get('status')!='PASS': raise RuntimeError('Historical comparison package did not seal correctly')\nprint(json.dumps(cert,indent=2,sort_keys=True))\nprint('Publish this output directory as an immutable private Kaggle Dataset, then attach that fixed version to the final Track-B notebook.')\n"""),
    ]
    return {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}


def main():
    out=Path(__file__).resolve().parents[1] / 'kaggle' / 'trackb_build_historical_compare.ipynb'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_notebook(),indent=1,ensure_ascii=False)+'\n',encoding='utf-8'); print(out)


if __name__=='__main__': main()
