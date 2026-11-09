"""Explicit current-cohort train/RAW/native-text entrypoints; no launch on import."""
import argparse,json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description='Arete native095 full-width RAW cohort; no pretrained qualification claimed')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('check')
    train=sub.add_parser('train');train.add_argument('--config',required=True);train.add_argument('--output',required=True)
    train.add_argument('--resume');train.add_argument('--stop-after',type=int)
    render=sub.add_parser('generate');render.add_argument('--artifact',required=True);render.add_argument('--local-options',required=True)
    render.add_argument('--request',required=True);render.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.command=='check':
        from .contracts import BACKEND,CORE_REVISION,SOURCE_PROFILE,INITIALIZATION_POLICY
        print(json.dumps(dict(backend=BACKEND,core=CORE_REVISION,PG='0.9.5',source_profile=SOURCE_PROFILE,
            initialization_policy=INITIALIZATION_POLICY,selection='raw',ema_decay=None,native_width=5120,
            placement='explicit single or two CUDA native module map; no auto/balanced',pretrained_qualified=False),indent=2));return
    if args.command=='train':
        from particle_sliders.lifecycle import SessionConfig
        from .lifecycle import open_training
        output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
        config=SessionConfig.from_dict(json.loads(Path(args.config).read_text()))
        with open_training(config,resume=args.resume)as session:
            def saved(metrics):session.checkpoint(output/f"FULL-{metrics['step']:06d}.pt")
            session.train(stop_after=args.stop_after,on_step=saved)
            session.export_inference(output/'RAW.pt',selection='raw')
    else:
        from particle_sliders.inference import open_inference
        from .lifecycle import INFERENCE
        request=json.loads(Path(args.request).read_text());options=json.loads(Path(args.local_options).read_text())
        strength=request.pop('strength',1.);seed=request.pop('seed',0)
        with open_inference(args.artifact,backend=INFERENCE,operation='render',options=options,
            strength=strength,output_type='text',native_request=request)as session:
            session.render(strength=strength,seed=seed,output_type='text',**request).save(args.output)

if __name__=='__main__':main()
