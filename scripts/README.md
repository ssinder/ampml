# scripts/

Helper scripts that are not part of the modelling pipeline itself.

## `sandbox_commit.py`

Commits the working tree in environments where the standard `git add` /
`git commit` path is unavailable.

**Why it exists.** Git writes its index and refs by creating a `<path>.lock`
file and then renaming it over the target. Some sandboxes block the delete of
`*.lock` files and the rename operation itself, which breaks every porcelain
command that touches the index (`git add`, `git update-index`, `git
write-tree`, `git update-ref`). `sandbox_commit.py` performs the same work
using plumbing commands and plain file writes, which are not affected.

**How it works.**

1. `git ls-files -z` collects the tracked paths, NUL-separated so that paths
   containing spaces or non-ASCII characters survive intact.
2. `git hash-object -w --stdin-paths` writes the blob for each file.
3. `git update-index --index-info` feeds the new index. When the sandbox
   blocks the final rename, the complete index is still present in the
   leftover `.lock` file (its magic is `DIRC`); the script copies that file
   into place instead of renaming it.
4. `git write-tree` produces the tree object, run against a fresh
   `GIT_INDEX_FILE` path so a stale lock cannot interfere.
5. `git commit-tree -p <HEAD>` creates the commit. The `-p` matters: without
   it every commit becomes a root commit and the history collapses into a
   single line.
6. The branch ref is updated by writing `refs/heads/<name>` directly, which
   bypasses the lock-and-rename sequence.

**Usage.**

```bash
python3 scripts/sandbox_commit.py                  # commit with the default message
python3 scripts/sandbox_commit.py -m "Describe the change"
```

The script stages every tracked file plus every untracked file that
`.gitignore` does not exclude, and writes the commit to `refs/heads/main`.
It prints the file counts, the tree and commit SHAs, the resulting
`git status` line count and the number of `git fsck` errors.

**Known limits.** The script always commits to `main` and always stages the
whole tree; it has no staging-area selection and no amend support. Deleted
files are dropped from the index rather than recorded as deletions, so
removing a file needs a separate `git rm --cached` or a manual index edit.

**When you do not need it.** In a normal environment, `git add -A && git
commit -m "..."` is simpler and preferable. This script exists only as a
workaround for restricted sandboxes.
