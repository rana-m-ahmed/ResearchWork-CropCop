import hashlib,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"journal_extension"/"src"))
from cropcop_je.data import ManifestColumns,load_manifest_rows,row_augmentation_seed
from cropcop_je.accumulation import accumulation_bucket_sample_count
from cropcop_je.runlog import assert_resume_identity,validate_run_record
from cropcop_je.surfaces import SurfaceAuthorizationError,authorize_training_surface,validate_training_config
from cropcop_je.validate import validate_static

class JECoreContractTests(unittest.TestCase):
    def load(self,p): return json.loads((ROOT/p).read_text())
    def test_static_g0_contract_passes(self):
        r=validate_static(ROOT); self.assertEqual(r["status"],"PASS",r["errors"])
    def test_exact_principal_ids_seeds_and_pairing(self):
        rows={r["experiment_id"]:r for r in self.load("journal_extension/locks/experiment_registry.json")["experiments"]}
        for i,seed in enumerate([21270083,606135704,1153870846],1):
            d=rows[f"R04-MNV4-DIRECT-S{i}"]; t=rows[f"R05-MNV4-TEACHER-S{i}"]
            self.assertEqual(d["seed"],seed); self.assertEqual(t["seed"],seed); self.assertEqual(d["pair_id"],t["pair_id"])
    def test_configs_fail_closed(self):
        for i in (1,2,3):
            d=self.load(f"journal_extension/configs/r04_direct/s{i}.json"); t=self.load(f"journal_extension/configs/r05_teacher/s{i}.json")
            validate_training_config(d); validate_training_config(t); self.assertEqual(d["required_student_init_evidence"],t["required_student_init_evidence"])
    def test_surface_guard(self):
        self.assertEqual(authorize_training_surface("DS-V1-TRAIN"),"DS-V1-TRAIN")
        with self.assertRaises(SurfaceAuthorizationError): authorize_training_surface("DS-V1-TEST-CONSUMED")
        with self.assertRaises(SurfaceAuthorizationError): authorize_training_surface("DS-EXT-POTATO-SEALED")
    def test_manifest_guard(self):
        text="rid,path,split,class_index\na,a.jpg,train,0\nb,b.jpg,val,1\nc,c.jpg,test,2\n"
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"m.csv"; p.write_text(text); sha=hashlib.sha256(p.read_bytes()).hexdigest(); cols=ManifestColumns("rid","path","split","class_index")
            self.assertEqual(len(load_manifest_rows(p,expected_sha256=sha,surface="DS-V1-TRAIN",columns=cols,expected_count=1)),1)
            with self.assertRaises(SurfaceAuthorizationError): load_manifest_rows(p,expected_sha256=sha,surface="DS-V1-TEST-CONSUMED",columns=cols)
    def test_deterministic_row_seed(self):
        a=row_augmentation_seed(21270083,0,"r"); self.assertEqual(a,row_augmentation_seed(21270083,0,"r")); self.assertNotEqual(a,row_augmentation_seed(21270083,1,"r"))
    def test_locked_training_tail_bucket_is_sample_normalized(self):
        self.assertEqual(
            accumulation_bucket_sample_count(
                absolute_batch_index=4772,
                batches_per_epoch=4774,
                micro_batch_size=16,
                accumulation_steps=4,
                dataset_size=76376,
            ),
            24,
        )
        self.assertEqual(
            accumulation_bucket_sample_count(
                absolute_batch_index=4773,
                batches_per_epoch=4774,
                micro_batch_size=16,
                accumulation_steps=4,
                dataset_size=76376,
            ),
            24,
        )

    def test_locked_validation_batch_is_64(self):
        ctc=self.load("journal_extension/configs/common/ctc_v2.json")
        self.assertEqual(ctc["training"]["validation_batch_size"],64)

    def test_teacher_targets_are_fp32_and_determinism_is_explicit(self):
        source=(ROOT/"journal_extension/src/cropcop_je/train.py").read_text()
        self.assertIn('torch.amp.autocast("cuda", enabled=False)',source)
        self.assertIn('kd_loss(sl.float(), tl.float()',source)
        self.assertIn('feature_loss(projection(sf).float(), tf.float())',source)
        self.assertIn('torch.backends.cudnn.benchmark = False',source)
        self.assertIn('torch.backends.cudnn.deterministic = True',source)
        self.assertIn('torch.use_deterministic_algorithms(True, warn_only=True)',source)

    def test_resume_drift_rejected(self):
        b={"experiment_id":"R04-MNV4-DIRECT-S1","authority_id":"EAAI-JE-SDL-v2.1-QA","config_sha256":"a"*64,"manifest_sha256":"b"*64,"class_map_sha256":"c"*64,"seed":21270083,"student_init_sha256":"d"*64,"pretrained_sha256":"e"*64,"teacher_sha256":None}
        assert_resume_identity(b,dict(b)); c=dict(b); c["seed"]=1
        with self.assertRaises(ValueError): assert_resume_identity(b,c)
    def test_run_record_rejects_test(self):
        r={"run_id":"x","experiment_id":"R04-MNV4-DIRECT-S1","authority_id":"EAAI-JE-SDL-v2.1-QA","source_git_commit":"0"*40,"config_sha256":"a"*64,"manifest_sha256":"b"*64,"class_map_sha256":"c"*64,"seed":21270083,"allowed_surfaces":["DS-V1-TRAIN","DS-V1-VAL"],"status":"LAUNCHED"}
        validate_run_record(r); r["allowed_surfaces"].append("DS-V1-TEST-CONSUMED")
        with self.assertRaises(ValueError): validate_run_record(r)
if __name__=="__main__": unittest.main()
