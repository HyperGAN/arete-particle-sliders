"""Run without importing Torch/Transformers or observing live resources."""
from pathlib import Path
import ast,contextlib,hashlib,importlib.util,io,json,sys,tempfile,unittest
N=Path(__file__).parents[1];R=N.parent
spec=importlib.util.spec_from_file_location('arete95_contract_stdlib',N/'src/arete_native95/contracts.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class ContractChecks(unittest.TestCase):
    def test_real_native5120_profile_and_exact_two_device_coverage(self):
        config=json.loads((R/'model-source/config.json').read_text())
        index=json.loads((R/'model-source/model.safetensors.index.json').read_text())
        placement=c.two_device_placement()
        self.assertEqual(c.validate_placement(placement,config),placement)
        self.assertEqual(config['text_config']['hidden_size'],5120)
        self.assertEqual(len(config['text_config']['layer_types']),64)
        self.assertEqual(sum(5 if k=='linear_attention'else 4 for k in config['text_config']['layer_types']),304)
        for name in index['weight_map']:
            matches=[key for key in placement['device_map']if name.startswith(key+'.')]
            self.assertEqual(len(matches),1)
        self.assertEqual(placement['owned_devices'],[{'device':'cuda:0'},{'device':'cuda:1'}])
        # These exact nested `device` keys are seen by current-core _rng_devices.
        self.assertEqual(placement['trainable_device'],'cuda:1')
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);(root/'config.json').write_text(json.dumps(config));(root/'rows.jsonl').write_text('{}\n{}\n')
            options=dict(model_dir=str(root),training_file=str(root/'rows.jsonl'),placement=placement,
                source_profile=c.SOURCE_PROFILE,initialization_policy=c.INITIALIZATION_POLICY,
                pg_version='0.9.5',pg_source_profile='release-0.9.5',penalty_input_domain='continuous',ema_decay=None)
            c.validate_native_options(options,training=True)
            for key,value in [('response_width',256),('ema_decay',.995),('placement',{'kind':'auto'}),('recipe_overrides',{'lr':.001})]:
                bad=dict(options,**{key:value})
                with self.assertRaises(ValueError):c.validate_native_options(bad,training=True)

    def test_raw_policy_common_literals_no_framework_or_files(self):
        literal={'lr':.001,'d_lr_mult':18.,'prior_lr_mult':30.}
        self.assertEqual(c.rate_overrides(literal),literal)
        for key in literal:
            for value in(True,False,0,-1,float('nan'),float('inf'),'0.001'):
                with self.assertRaises(ValueError):c.rate_overrides({key:value})
        for value in(.995,False,0.):
            with self.assertRaises(ValueError):c.validate_profile(dict(ema_decay=value,source_profile=c.SOURCE_PROFILE,
                initialization_policy=c.INITIALIZATION_POLICY,pg_version='0.9.5',pg_source_profile='release-0.9.5',penalty_input_domain='continuous'))
        self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)

    def test_native_source_extraction_and_all_new_syntax(self):
        host=N/'src/arete_native95/host_source.py';old=R/'src/arete_particle_sliders/backend.py'
        self.assertEqual(host.read_bytes(),old.read_bytes())
        identity=json.loads((host.parent/'host-source.json').read_text())
        self.assertEqual(hashlib.sha256(host.read_bytes()).hexdigest(),identity['sha256'])
        self.assertEqual((host.parent/'model.lock.json').read_bytes(),(old.parent/'model.lock.json').read_bytes())
        for path in N.rglob('*.py'):ast.parse(path.read_text())

    def test_current_check_cli_is_allocation_free(self):
        sys.path.insert(0,str(N/'src'))
        from arete_native95.cli import main
        out=io.StringIO()
        with contextlib.redirect_stdout(out):main(['check'])
        value=json.loads(out.getvalue())
        self.assertEqual(value['native_width'],5120);self.assertEqual(value['PG'],'0.9.5');self.assertIsNone(value['ema_decay'])
        self.assertFalse(value['pretrained_qualified']);self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main()
