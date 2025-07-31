# DDoS Log Analysis

**ID:** `ddos_log_analysis`

**Category:** data_processing

**Description:** Security log analysis with temporal spike pattern (DDoS simulation)

## Dependencies

- python3
- gzip
- awk

## Small Setup Commands

```bash
cp $ORIGINAL_CWD/assets/pandas_etl_prepare.py $ORIGINAL_CWD/assets/pandas_etl_test.py .
cp pandas_etl_test.py small_pandas_etl_test.py
chmod +x pandas_etl_prepare.py small_pandas_etl_test.py
python3 pandas_etl_prepare.py --normal-logs 30000 --error-logs 8000 --output etl_small.gz
```

## Large Setup Commands

```bash
cp $ORIGINAL_CWD/assets/pandas_etl_prepare.py $ORIGINAL_CWD/assets/pandas_etl_test.py .
cp pandas_etl_test.py large_pandas_etl_test.py
chmod +x pandas_etl_prepare.py large_pandas_etl_test.py
python3 pandas_etl_prepare.py --normal-logs 1000000 --error-logs 500000 --output etl_large.gz
```

## Small Execution Commands

```bash
./small_pandas_etl_test.py etl_small.gz
```

## Large Execution Commands

```bash
./large_pandas_etl_test.py etl_large.gz
```

## Cleanup Commands

```bash
rm -f etl_small.gz etl_large.gz
rm -f /tmp/pandas_*
rm -f small_pandas_etl_test.py large_pandas_etl_test.py pandas_etl_prepare.py pandas_etl_test.py
```
