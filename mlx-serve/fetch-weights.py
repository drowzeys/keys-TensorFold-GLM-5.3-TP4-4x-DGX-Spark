"""fetch-weights.py REPO DIR [VERIFY]: a public Hugging Face model repository into DIR with curl, resumable.

Runs on a Spark (install.sh pipes it over ssh). A file that already has the repository's size is kept; a partial one
is continued. VERIFY=1 also compares the sha256 of every large file with the repository's. Needs python3 and curl.
A gated repository needs a token: HF_TOKEN in the environment, or the one `hf auth login` saved on this machine.
FETCH_MAX_MB in the environment leaves larger files out (a test of the path; the result is then not a checkpoint)."""
import hashlib, json, os, subprocess, sys, urllib.error, urllib.request

repo, dest, verify = sys.argv[1], sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == "1"
token = os.environ.get("HF_TOKEN", "")
for saved in ("~/.cache/huggingface/token", "~/.huggingface/token"):
    if not token and os.path.isfile(os.path.expanduser(saved)):
        token = open(os.path.expanduser(saved)).read().strip()
auth = {"Authorization": "Bearer " + token} if token else {}
try:
    api = json.load(urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/models/{repo}?blobs=true", headers=auth), timeout=60))
except urllib.error.HTTPError as error:
    sys.exit(f"FAILED {repo}: HTTP {error.code}" + (": a gated or private repository; accept its terms on huggingface.co and give a token (HF_TOKEN, or hf auth login on each Spark)" if error.code in (401, 403, 404) else ""))
files = [(s["rfilename"], s.get("size") or 0, (s.get("lfs") or {}).get("sha256")) for s in api["siblings"]]
limit = int(os.environ.get("FETCH_MAX_MB", 0)) << 20
whole = len(files)
if limit:
    files = [f for f in files if f[1] <= limit]
total = sum(size for _, size, _ in files)
print(f"{repo} at {api['sha'][:12]}: {len(files)} files, {total / 1e9:.1f} GB -> {dest}", flush=True)
have = 0
for i, (name, size, sha) in enumerate(sorted(files, key=lambda f: f[1])):
    path = os.path.join(dest, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not (os.path.isfile(path) and os.path.getsize(path) == size):
        if os.path.isfile(path) and os.path.getsize(path) > size:
            os.remove(path)
        url = f"https://huggingface.co/{repo}/resolve/{api['sha']}/{name}"
        # the token goes to curl in a config on its standard input, so it is not on a command line
        config = f'header = "Authorization: Bearer {token}"\n' if token else ""
        done = subprocess.run(["curl", "-fsSL", "--retry", "8", "--retry-delay", "5", "--retry-all-errors", "-C", "-", "-K", "-", "-o", path, url], input=config.encode())
        if done.returncode or os.path.getsize(path) != size:
            hint = "run again to resume" if token or done.returncode != 22 else "the server refused it: a gated repository needs its terms accepted on huggingface.co and a token (HF_TOKEN, or hf auth login on each Spark)"
            sys.exit(f"FAILED {name}: curl exit {done.returncode}, {os.path.getsize(path) if os.path.isfile(path) else 0} of {size} bytes ({hint})")
    if verify and sha:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while chunk := handle.read(1 << 24):
                digest.update(chunk)
        if digest.hexdigest() != sha:
            os.rename(path, path + ".bad")
            sys.exit(f"FAILED {name}: sha256 differs from the repository's (kept as {name}.bad; run again)")
    have += size
    print(f"{i + 1}/{len(files)} {name}  {have / 1e9:.1f} of {total / 1e9:.1f} GB", flush=True)
if len(files) != whole:
    sys.exit(f"PARTIAL {len(files)} of {whole} files (FETCH_MAX_MB is set)")
print(f"WEIGHTS OK {len(files)} files, {total / 1e9:.1f} GB" + (", sha256 checked" if verify else ""), flush=True)
