"""sync_and_push: local media commit must land on origin even after the remote advanced (PR merged elsewhere)."""
import importlib.util
import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ig_publish", os.path.join(HERE, "..", "tools", "ig_publish.py"))
igp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(igp)


def git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class SyncAndPushTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.origin = os.path.join(t, "origin.git")
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", self.origin], check=True)
        self.local, self.other = os.path.join(t, "local"), os.path.join(t, "other")
        for d in (self.local, self.other):
            subprocess.run(["git", "clone", "-q", self.origin, d], check=True, capture_output=True)
            git(d, "config", "user.email", "t@t"); git(d, "config", "user.name", "t")
        open(os.path.join(self.local, "a.txt"), "w").write("a")
        git(self.local, "add", "a.txt"); git(self.local, "commit", "-qm", "init"); git(self.local, "push", "-q", "origin", "HEAD:main")
        git(self.other, "pull", "-q", "origin", "main")

    def tearDown(self):
        self.tmp.cleanup()

    def test_remote_advanced_rebases_and_pushes(self):
        open(os.path.join(self.other, "pr.py"), "w").write("pr")
        git(self.other, "add", "pr.py"); git(self.other, "commit", "-qm", "PR"); git(self.other, "push", "-q", "origin", "HEAD:main")
        open(os.path.join(self.local, "reel.mp4"), "w").write("mp4")
        git(self.local, "add", "reel.mp4"); git(self.local, "commit", "-qm", "media")
        igp.sync_and_push(self.local)
        files = git(self.origin, "ls-tree", "--name-only", "main").split()
        self.assertIn("pr.py", files)
        self.assertIn("reel.mp4", files)
        self.assertEqual(git(self.origin, "rev-list", "--count", "main"), "3")  # linear, nothing overwritten

    def test_conflict_aborts_without_force(self):
        open(os.path.join(self.other, "a.txt"), "w").write("remote")
        git(self.other, "commit", "-qam", "remote edit"); git(self.other, "push", "-q", "origin", "HEAD:main")
        remote_head = git(self.origin, "rev-parse", "main")
        open(os.path.join(self.local, "a.txt"), "w").write("local")
        git(self.local, "commit", "-qam", "local edit")
        with self.assertRaises(RuntimeError):
            igp.sync_and_push(self.local)
        self.assertEqual(git(self.origin, "rev-parse", "main"), remote_head)  # remote untouched
        self.assertFalse(os.path.exists(os.path.join(self.local, ".git", "rebase-merge")))


if __name__ == "__main__":
    unittest.main()
