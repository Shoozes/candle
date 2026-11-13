import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

from prepare_packet import source_inventory


class SourceInventoryTests(unittest.TestCase):
    def test_shared_worktree_changes_are_bound_to_the_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()

            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

            def write(name, data):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)

            def digest(name):
                return hashlib.sha256((root / name).read_bytes()).hexdigest()

            kernel = "candle-core/src/quantized/repack_x86.rs"
            staged = "candle-core/src/deleted_from_index.rs"
            unstaged = "candle-nn/src/deleted_from_worktree.rs"
            added = "candle-core/src/local_kernel.rs"
            status = "docs/lfm2-moe/STATUS.md"
            git("init", "-q")
            tracked = (kernel, staged, unstaged, "Cargo.lock", status)
            for name in tracked:
                write(name, b"original")
            git("add", "--", *tracked)
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
            head = git("rev-parse", "HEAD")

            write(kernel, b"local arithmetic")
            git("rm", "-q", "--", staged)
            (root / unstaged).unlink()
            write(added, b"new source")
            write("candle-core/src/scratch.tmp", b"local scratch")
            write("artifacts/model-free-scratch.rs", b"generated output")

            rows = source_inventory(root, head, [status, "Cargo.lock"])
            inventory = {row["path"]: row["sha256"] for row in rows}
            self.assertEqual(list(inventory), sorted(inventory))
            self.assertEqual(len(rows), len(inventory))
            self.assertEqual(inventory, {
                kernel: digest(kernel),
                staged: None,
                unstaged: None,
                added: digest(added),
                status: digest(status),
                "Cargo.lock": digest("Cargo.lock"),
            })


if __name__ == "__main__":
    unittest.main()
