"""Excel-readable CSV views of authoritative JSON results (AP units: 0–100)."""
import argparse
import csv
import json
import warnings
from pathlib import Path


def flatten(row, prefix=''):
    result = {}
    for key, value in row.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        elif isinstance(value, (list, tuple)):
            result[name] = json.dumps(value, ensure_ascii=False)
        else:
            result[name] = value
    return result


def write_csv(path, rows):
    """Rebuild atomically so resume cannot append duplicate rounds."""
    path = Path(path)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    except PermissionError:
        warnings.warn(f'Cannot update {path}; close it in Excel and export again. JSON results are preserved.',
                      RuntimeWarning)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_training_tables(folder):
    folder = Path(folder)
    log = folder / 'rounds.jsonl'
    if not log.exists():
        return
    records = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records:
        return
    cfg = json.loads((folder / 'config.json').read_text(encoding='utf-8-sig'))
    rows = []
    for record in records:
        row = {'method': cfg['method'], 'seed': cfg['seed'], 'split': 'dev', 'AP_unit': '0-100'}
        row.update(flatten(record))
        row['round'] = record['round'] + 1
        rows.append(row)
    write_csv(folder / 'rounds.csv', rows)
    summary = [{'selection': 'last', **rows[-1]}]
    evaluated = [i for i, r in enumerate(records)
                 if r.get('dev') is not None and r['dev'].get('AP50_95') is not None]
    if evaluated:
        best = max(evaluated, key=lambda i: records[i]['dev']['AP50_95'])
        summary.append({'selection': 'best_dev', **rows[best]})
    write_csv(folder / 'summary.csv', summary)


def export_final_table(path, result):
    row = {'split': 'test', 'AP_unit': '0-100', **flatten(result)}
    if 'round' in row:
        row['round'] += 1
    write_csv(Path(path).with_suffix('.csv'), [row])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    export_training_tables(args.run_dir)
    for path in args.run_dir.glob('*.final.json'):
        export_final_table(path, json.loads(path.read_text(encoding='utf-8')))
    print(args.run_dir)


if __name__ == '__main__':
    main()
