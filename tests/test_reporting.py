import csv
import json

from feddecouple.reporting import export_training_tables, export_final_table


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def test_round_tables_preserve_missing_metrics_and_rebuild_after_resume(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'method': 'fedavg', 'seed': 42}))
    records = [
        {'round': 0, 'loss': 0., 'dev': {'AP50_95': 2., 'per_class': {'car': 3.}}},
        {'round': 1, 'loss': 1., 'dev': None},
        {'round': 2, 'loss': 2., 'dev': {'AP50_95': 1., 'AP_small': None}},
    ]
    log = tmp_path / 'rounds.jsonl'
    log.write_text(''.join(json.dumps(r) + '\n' for r in records))
    export_training_tables(tmp_path)
    rows = read_csv(tmp_path / 'rounds.csv')
    assert len(rows) == 3
    assert rows[0]['round'] == '1' and rows[0]['loss'] == '0.0'
    assert rows[0]['dev.per_class.car'] == '3.0'
    assert rows[1]['dev.AP50_95'] == '' and rows[2]['dev.AP_small'] == ''
    summary = read_csv(tmp_path / 'summary.csv')
    assert [(r['selection'], r['round']) for r in summary] == [('last', '3'), ('best_dev', '1')]
    # Simulate resume truncating an uncommitted round, then refresh the derived CSV.
    log.write_text(json.dumps(records[0]) + '\n')
    export_training_tables(tmp_path)
    assert len(read_csv(tmp_path / 'rounds.csv')) == 1


def test_final_table_identifies_test_split_and_preserves_scale(tmp_path):
    path = tmp_path / 'last.final.json'
    export_final_table(path, {'round': 1, 'AP50_95': 25.5, 'AP_small': None})
    row, = read_csv(tmp_path / 'last.final.csv')
    assert row['split'] == 'test' and row['AP_unit'] == '0-100'
    assert row['AP50_95'] == '25.5' and row['AP_small'] == ''
    assert row['round'] == '2'
