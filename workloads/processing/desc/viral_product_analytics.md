# Viral Product Analytics

**ID:** `viral_product_analytics`

**Category:** data_processing

**Description:** Retail analytics with temporal hot product pattern (trending item simulation)

## Dependencies

- python3
- awk

## Small Setup Commands

```bash
cp $ORIGINAL_CWD/assets/flink_join_prepare.py $ORIGINAL_CWD/assets/flink_join_test.py .
cp flink_join_test.py small_flink_join_test.py
chmod +x flink_join_prepare.py small_flink_join_test.py
python3 flink_join_prepare.py --regular-transactions 8000 --hot-transactions 12000 --output flink_small.csv
```

## Large Setup Commands

```bash
cp $ORIGINAL_CWD/assets/flink_join_prepare.py $ORIGINAL_CWD/assets/flink_join_test.py .
cp flink_join_test.py large_flink_join_test.py
chmod +x flink_join_prepare.py large_flink_join_test.py
python3 flink_join_prepare.py --regular-transactions 100000 --hot-transactions 700000 --output flink_large.csv
```

## Small Execution Commands

```bash
./small_flink_join_test.py flink_small.csv
```

## Large Execution Commands

```bash
./large_flink_join_test.py flink_large.csv
```

## Cleanup Commands

```bash
rm -f flink_small.csv flink_large.csv
rm -f /tmp/flink_*
rm -f small_flink_join_test.py large_flink_join_test.py flink_join_prepare.py flink_join_test.py
```
