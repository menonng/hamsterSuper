"""Publish trained models as GitHub Releases.

Methods (``method="auto"`` picks the first that is available):
  * ``gh``      - the `gh` CLI, when installed and authenticated
  * ``api``     - the GitHub REST API with $GITHUB_TOKEN (or $GH_TOKEN)
  * ``actions`` - needs only `git push`: a commit with the model files under
                  release/ is tagged and the tag pushed; the workflow
                  .github/workflows/model-release.yml then creates the release.
The target repository is read from `git remote get-url origin` unless given.
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
    def __init__(self, repo: str | None = None, cwd=None, dry_run: bool = False, target: str | None = None,
                 method: str = "auto", remote: str = "origin"):
        self.cwd = cwd
        self.method, self.remote = method, remote
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
        method = self.method
        if method == "auto":
            method = "gh" if shutil.which("gh") and self._gh_ok() else ("api" if self.token else "actions")
        if method == "gh":
            url = self._publish_gh(tag, title, notes, files)
        elif method == "api":
            if not self.token:
                raise RuntimeError("method 'api' needs GITHUB_TOKEN")
            url = self._publish_rest(tag, title, notes, files)
        else:
            url = self._publish_actions(tag, title, notes, files)
        msg = f"[release] published {tag} -> {url}"
        print(msg, flush=True)
        self.log.append(msg)
        return url

    def _publish_actions(self, tag, title, notes, files):
        """Commit HEAD's tree + release/<files> + .release-notes.md (parent HEAD),
        tag it and push only the tag. Uses a temporary index, so the working
        tree and branches are untouched."""
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(_git("rev-parse", "--git-dir", cwd=self.cwd)).resolve()
                                                   / f"index-release-{tag}")}

        def g(*args, inp=None):
            return subprocess.run(["git", *args], cwd=self.cwd, env=env, check=True, capture_output=True,
                                  text=inp is None or isinstance(inp, str), input=inp).stdout

        try:
            g("read-tree", "HEAD")
            entries = [(f, f"release/{Path(f).name}") for f in files]
            notes_file = Path(env["GIT_INDEX_FILE"] + ".md")
            notes_file.write_text(f"# {title}\n\n{notes}", encoding="utf-8")
            entries.append((str(notes_file), ".release-notes.md"))
            for src, dst in entries:
                sha = g("hash-object", "-w", src).strip()
                g("update-index", "--add", "--cacheinfo", f"100644,{sha},{dst}")
            tree = g("write-tree").strip()
            commit = g("commit-tree", tree, "-p", "HEAD", "-m", f"Release {tag}").strip()
            g("tag", "-f", tag, commit)
            res = subprocess.run(["git", "push", "-f", self.remote, f"refs/tags/{tag}"], cwd=self.cwd,
                                 capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(res.stderr.strip() or res.stdout.strip())
        finally:
            for p in (Path(env["GIT_INDEX_FILE"]), Path(env["GIT_INDEX_FILE"] + ".md")):
                p.unlink(missing_ok=True)
        return f"https://github.com/{self.repo}/releases/tag/{tag} (created by the 'Model release' workflow)"

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


def release_notes(metrics: dict) -> str:
    """Markdown release notes from a release's metrics.json."""
    tag = metrics.get("tag", "")
    er = metrics.get("eval_random", {})
    t = er.get("mean_time_s")
    notes = (f"## HamsterSuper autonomous-driving policy `{tag}`\n\n"
             f"- PPO iterations: {metrics.get('iter')} / episodes: {metrics.get('episodes')}\n"
             f"- Curriculum level: {metrics.get('curriculum_level')}\n")
    if er:
        notes += (f"- Random maps (hardest level, domain randomization ON): success rate "
                  f"**{er['success_rate']:.1%}**, collisions per episode {er['collisions_per_episode']:.2f}, "
                  f"mean finish time {'-' if t is None else round(t, 2)}s\n")
    if metrics.get("eval_maps"):
        notes += f"- Example maps: success rate **{metrics['eval_maps']['success_rate']:.1%}**\n"
    notes += ("\n### Files\n- `policy_ts.pt`: TorchScript model for the real robot/server "
              "(`python -m hamster_super.server --model policy_ts.pt`)\n"
              "- `policy.npz`: the same policy for NumPy-only inference (no torch, e.g. Raspberry Pi)\n"
              "- `checkpoint.pt`: resume training (`--resume`)\n- `metrics.json`, `env_config.json`\n")
    return notes


def main():
    """Publish an existing release folder (runs/<run>/releases/<tag>)."""
    import argparse
    ap = argparse.ArgumentParser(description="Publish saved model releases to GitHub")
    ap.add_argument("dirs", nargs="+", help="release folders containing metrics.json")
    ap.add_argument("--method", choices=["auto", "gh", "api", "actions"], default="auto")
    ap.add_argument("--repo", default=None)
    args = ap.parse_args()
    rel = GitHubReleaser(repo=args.repo, method=args.method)
    for d in map(Path, args.dirs):
        m = json.loads((d / "metrics.json").read_text())
        tag = m.get("tag", d.name)
        files = [d / n for n in ("policy_ts.pt", "policy.npz", "checkpoint.pt", "metrics.json", "env_config.json")]
        rel.publish(tag, f"HamsterSuper {tag}", release_notes(m), files, background=False)


if __name__ == "__main__":
    main()
