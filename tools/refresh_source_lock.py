"""Refresh product hashes while checking the immutable inherited core closure."""
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'src/arete_particle_sliders'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sources(root):
    return {str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def main():
    lock_path = ROOT / 'sources.lock.json'
    lock = json.loads(lock_path.read_text())
    dist = distribution('particle-sliders-core')
    origin = json.loads(dist.read_text('direct_url.json') or '{}')
    if (dist.version != lock['core_version']
            or origin.get('vcs_info', {}).get('commit_id') != lock['core_commit']
            or origin.get('url') != lock['core_repository']
            or origin.get('subdirectory') != lock['core_subdirectory']):
        raise SystemExit('immutable installed core origin differs')
    import particle_sliders
    roots = {'particle_sliders': Path(particle_sliders.__file__).parent,
             'concept_slider_core': Path(dist.locate_file('concept_slider_core'))}
    actual = {package + '/' + path: value for package, root in roots.items()
              for path, value in sources(root).items()}
    if actual != lock['core_sources']:
        raise SystemExit('inherited core source closure differs; do not rebaseline')
    product = sources(ROOT)
    del product['sources.lock.json']
    lock['product_sources'] = product
    lock['product_sources_index_sha256'] = digest(json.dumps(
        product, sort_keys=True, separators=(',', ':')).encode())
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(core_files=len(actual), product_files=len(product))))


if __name__ == '__main__':
    main()
