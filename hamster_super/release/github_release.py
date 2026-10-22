"""Publish trained models as GitHub Releases.

Uses the `gh` CLI when it is installed and authenticated, otherwise the
GitHub REST API with a token from $GITHUB_TOKEN (or $GH_TOKEN). The target
repository is read from `git remote get-url origin` unless given.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import urllib.parse
import urllib.request
from pathlib import Path


def _git(*args, cwd=None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def detect_repo(cwd=None) -> str:
    url = _git("remote", "get-url", "origin", cwd=cwd)
    m = re.search(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if not m:
        # proxied remotes such as http://host/git/owner/repo
        m = re.search(r"/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if not m:
        raise RuntimeError(f"cannot infer owner/repo from remote {url!r}")
    return f"{m.group(1)}/{m.group(2)}"


class GitHubReleaser:
    def __init__(self, repo: str | None = None, cwd=None, dry_run: bool = False, target: str | None = None):
        self.cwd = cwd
        self.dry_run = dry_run
        self.repo = repo or detect_repo(cwd)
        self.target = target or self._pushed_head()
        self.token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        self._threads: list[threading.Thread] = []
        self.log: list[str] = []

    def _pushed_head(self):
        """Tag the current commit only if it exists on the remote; otherwise
        let GitHub use the default branch."""
        try:
            sha = _git("rev-parse", "HEAD", cwd=self.cwd)
            return sha if _git("branch", "-r", "--contains", sha, cwd=self.cwd) else None
        except subprocess.CalledProcessError:
            return None

    # ------------------------------------------------------------------ API
    def publish(self, tag: str, title: str, notes: str, files: list, background: bool = True):
        if background:
            t = threading.Thread(target=self._publish_safe, args=(tag, title, notes, files), daemon=False)
            t.start()
            self._threads.append(t)
            return t
        return self._publish(tag, title, notes, files)

    def wait(self):
        for t in self._threads:
            t.join()

    # ------------------------------------------------------------- internals
    def _publish_safe(self, *a):
        try:
            self._publish(*a)
        except Exception as e:  # never kill training because a release failed
            msg = f"[release] FAILED {a[0]}: {e}"
            print(msg, flush=True)
            self.log.append(msg)

    def _publish(self, tag, title, notes, files):
        files = [str(Path(f)) for f in files if Path(f).exists()]
        if self.dry_run:
            msg = f"[release] dry-run {self.repo} {tag}: {', '.join(Path(f).name for f in files)}"
            print(msg, flush=True)
            self.log.append(msg)
            return None
        if shutil.which("gh") and self._gh_ok():
            url = self._publish_gh(tag, title, notes, files)
        elif self.token:
            url = self._publish_rest(tag, title, notes, files)
        else:
            raise RuntimeError("no GitHub credentials: run `gh auth login` or set GITHUB_TOKEN")
        msg = f"[release] published {tag} -> {url}"
        print(msg, flush=True)
        self.log.append(msg)
        return url

    def _gh_ok(self) -> bool:
        return subprocess.run(["gh", "auth", "status"], capture_output=True).returncode == 0

    def _publish_gh(self, tag, title, notes, files):
        cmd = ["gh", "release", "create", tag, *files, "--repo", self.repo, "--title", title, "--notes", notes]
        if self.target:
            cmd += ["--target", self.target]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=self.cwd)
        return out.stdout.strip()

    def _api(self, method, url, data=None, headers=None, raw=False):
        h = {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
             "X-GitHub-Api-Version": "2022-11-28"}
        h.update(headers or {})
        body = data if raw else (json.dumps(data).encode() if data is not None else None)
        req = urllib.request.Request(url, data=body, method=method, headers=h)
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read() or b"{}")

    def _publish_rest(self, tag, title, notes, files):
        payload = {"tag_name": tag, "name": title, "body": notes}
        if self.target:
            payload["target_commitish"] = self.target
        rel = self._api("POST", f"https://api.github.com/repos/{self.repo}/releases", payload)
        upload = rel["upload_url"].split("{")[0]
        for f in files:
            name = Path(f).name
            ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
            self._api("POST", f"{upload}?name={urllib.parse.quote(name)}", Path(f).read_bytes(),
                      headers={"Content-Type": ctype}, raw=True)
        return rel.get("html_url")
