# Git Add Different Size Directories

**ID:** `git_add_different`

**Category:** version_control

**Description:** Git add operations with different numbers of files

## Dependencies

- git
- g++

## Small Setup Commands

```bash
mkdir -p small_repo && cd small_repo && git init
cd small_repo && git config user.name 'Test User' && git config user.email 'test@example.com'
cd small_repo && dd if=/dev/urandom of=large_file_1.bin bs=20M count=1 2>/dev/null
cd small_repo && dd if=/dev/urandom of=large_file_2.bin bs=20M count=1 2>/dev/null
cd small_repo && mkdir -p src && for i in {1..100}; do echo "// File $i" > src/file_$i.js; done
cp $ORIGINAL_CWD/assets/git_add_libgit2.cpp .
g++ -o small_git_add git_add_libgit2.cpp -lgit2
```

## Large Setup Commands

```bash
mkdir -p large_repo && cd large_repo && git init
cd large_repo && git config user.name 'Test User' && git config user.email 'test@example.com'
cd large_repo && dd if=/dev/urandom of=huge_file_1.bin bs=100M count=1 2>/dev/null
cd large_repo && dd if=/dev/urandom of=huge_file_2.bin bs=100M count=1 2>/dev/null
cd large_repo && mkdir -p src && for i in {1..500}; do dd if=/dev/urandom of=src/file_$i.dat bs=1M count=1 2>/dev/null; done
cp $ORIGINAL_CWD/assets/git_add_libgit2.cpp .
g++ -o large_git_add git_add_libgit2.cpp -lgit2
```

## Small Execution Commands

```bash
./small_git_add small_repo
```

## Large Execution Commands

```bash
./large_git_add large_repo
```

## Cleanup Commands

```bash
rm -rf small_repo/ large_repo/
```
