# Hot Key Aggregation

**ID:** `hotkey_aggregation`

**Category:** data_processing

**Description:** Analytics with skewed data distribution (hot key problem)

## Dependencies

- python3

## Small Setup Commands

```bash
cp $ORIGINAL_CWD/assets/spark_skew_prepare.py $ORIGINAL_CWD/assets/spark_skew_test.py .
cp spark_skew_test.py small_spark_skew_test.py
chmod +x spark_skew_prepare.py small_spark_skew_test.py
python3 spark_skew_prepare.py --regular-keys 8000 --hot-keys 15000 --output spark_small.csv
```

## Large Setup Commands

```bash
cp $ORIGINAL_CWD/assets/spark_skew_prepare.py $ORIGINAL_CWD/assets/spark_skew_test.py .
cp spark_skew_test.py large_spark_skew_test.py
chmod +x spark_skew_prepare.py large_spark_skew_test.py
python3 spark_skew_prepare.py --regular-keys 100000 --hot-keys 800000 --output spark_large.csv
```

## Small Execution Commands

```bash
./small_spark_skew_test.py spark_small.csv
```

## Large Execution Commands

```bash
./large_spark_skew_test.py spark_large.csv
```

## Cleanup Commands

```bash
rm -f spark_small.csv spark_large.csv
rm -f small_spark_skew_test.py large_spark_skew_test.py spark_skew_prepare.py spark_skew_test.py
```
