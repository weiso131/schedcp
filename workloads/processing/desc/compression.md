# File Compression

**ID:** `compression`

**Category:** file_processing

**Description:** Compression of mixed-size files with severe load imbalance

## Dependencies

- python3

## Small Setup Commands

```bash
mkdir -p test_data
seq 1 3000000 > test_data/short_file.dat
cp $ORIGINAL_CWD/assets/compression.py .
cp compression.py small_compression.py
chmod +x small_compression.py
```

## Large Setup Commands

```bash
mkdir -p test_data
seq 1 20000000 > test_data/large_file.dat
cp $ORIGINAL_CWD/assets/compression.py .
cp compression.py large_compression.py
chmod +x large_compression.py
```

## Small Execution Commands

```bash
./small_compression.py test_data/short_file.dat 9
```

## Large Execution Commands

```bash
./large_compression.py test_data/large_file.dat 9
```

## Cleanup Commands

```bash
rm -rf test_data/
rm -f *.gz
rm -f small_compression.py large_compression.py compression.py
```
