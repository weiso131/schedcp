# Log Processing with Skewed Chunks

**ID:** `log_processing`

**Category:** data_processing

**Description:** Processing log files with different sizes and compression

## Dependencies

- gzip
- grep
- awk
- sort
- uniq

## Small Setup Commands

```bash
mkdir -p log_chunks
seq 1 500000 | awk '{print strftime("%Y-%m-%d %H:%M:%S"), "[INFO]", "Request from", "192.168.1."int(rand()*255), "processed in", int(rand()*100), "ms"}' > log_chunks/small.log
```

## Large Setup Commands

```bash
mkdir -p log_chunks
seq 1 7000000 | awk '{print strftime("%Y-%m-%d %H:%M:%S"), "[INFO]", "Request from", "192.168.1."int(rand()*255), "processed in", int(rand()*100), "ms"}' > log_chunks/large.log
```

## Small Execution Commands

```bash
gzip -c log_chunks/small.log | zcat | grep -E '\[INFO\]' | awk '{print $4}' | sort | uniq -c | sort -nr > log_chunks/small_ips.txt
```

## Large Execution Commands

```bash
gzip -c log_chunks/large.log | zcat | grep -E '\[INFO\]' | awk '{print $4}' | sort | uniq -c | sort -nr > log_chunks/large_ips.txt
```

## Cleanup Commands

```bash
rm -rf log_chunks/
```
