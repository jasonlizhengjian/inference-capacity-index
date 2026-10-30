"""Verify archived study bytes and table rows without importing a simulator."""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1] / 'docs/research/architecture-studies'


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    members = manifest['archive_members']
    checked = set()
    rows = 0
    for name, record in manifest['files'].items():
        path = ROOT / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], name
        if path.suffix != '.zip':
            continue
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                assert not Path(member).is_absolute() and '..' not in Path(member).parts
                key = path.parent.name + '/' + member
                record = members[key]
                data = archive.read(member)
                assert hashlib.sha256(data).hexdigest() == record['sha256'], key
                if record['data_rows'] is not None:
                    count = sum(1 for _ in csv.reader(io.StringIO(data.decode()))) - 1
                    assert count == record['data_rows'], key
                    rows += count
                checked.add(key)
    assert checked == set(members)
    print(json.dumps(dict(status='passed', files=len(manifest['files']),
                          archive_members=len(checked), csv_rows=rows,
                          simulations_run=False)))


if __name__ == '__main__':
    main()
