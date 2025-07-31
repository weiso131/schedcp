# CTest Suite with Slow Integration Test

**ID:** `ctest_suite`

**Category:** software_testing

**Description:** Test suite with fast unit tests and one slow integration test

## Dependencies

- gcc

## Small Setup Commands

```bash
cp $ORIGINAL_CWD/assets/short.c .
gcc -O2 short.c -lm -o short
```

## Large Setup Commands

```bash
cp $ORIGINAL_CWD/assets/long.c .
gcc -O2 long.c -lm -o long
```

## Small Execution Commands

```bash
./short
```

## Large Execution Commands

```bash
./long
```

## Cleanup Commands

```bash
rm -f assets/short assets/long
```
