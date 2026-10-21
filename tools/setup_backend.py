"""Download the pinned official backend, without executing external setup scripts."""
import io
from pathlib import Path
import urllib.request
import zipfile

COMMIT = 'df2a96c8c3a098567a1149bf0248da6efe6d495d'
dest = Path('third_party/RT-DETR').resolve()
if dest.exists():
    raise SystemExit(f'{dest} already exists; leave existing checkout untouched')
url = f'https://codeload.github.com/lyuwenyu/RT-DETR/zip/{COMMIT}'
data = urllib.request.urlopen(url, timeout=120).read()
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    for entry in archive.infolist():
        relative = Path(*Path(entry.filename).parts[1:])
        if not relative.parts or entry.is_dir():
            continue
        if relative.parts[0] != 'rtdetr_pytorch' and relative.name not in ('LICENSE', 'README.md'):
            continue
        target = (dest/relative).resolve()
        if not target.is_relative_to(dest):
            raise ValueError('Unsafe archive path')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(entry))
(dest/'COMMIT').write_text(COMMIT, encoding='utf8')
print(dest)
