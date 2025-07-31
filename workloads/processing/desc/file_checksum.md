# Parallel File System Operations

**ID:** `file_checksum`

**Category:** file_processing

**Description:** Checksum operations with one large file blocking completion

## Dependencies

- python3

## Small Setup Commands

```bash
mkdir -p large-dir
dd if=/dev/urandom of=large-dir/short_file.dat bs=1M count=200 2>/dev/null
cp $ORIGINAL_CWD/assets/file_checksum.py .
cp file_checksum.py small_file_checksum.py
chmod +x small_file_checksum.py
```

## Large Setup Commands

```bash
mkdir -p large-dir
dd if=/dev/urandom of=large-dir/long_file.dat bs=1M count=1000 2>/dev/null
cp $ORIGINAL_CWD/assets/file_checksum.py .
cp file_checksum.py large_file_checksum.py
chmod +x large_file_checksum.py
```

## Small Execution Commands

```bash
./small_file_checksum.py large-dir/short_file.dat
```

## Large Execution Commands

```bash
./large_file_checksum.py large-dir/long_file.dat
```

## Cleanup Commands

```bash
rm -rf large-dir/
rm -f checksums.txt
rm -f small_file_checksum.py large_file_checksum.py file_checksum.py
```
