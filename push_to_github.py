# -*- coding: utf-8 -*-
import subprocess, json, sys, urllib.request, urllib.error

# Get token via git credential (never printed to output)
proc = subprocess.run(
    ['git', 'credential', 'fill'],
    input='protocol=https\nhost=github.com\n',
    capture_output=True, text=True
)
token = None
for line in proc.stdout.splitlines():
    if line.startswith('password='):
        token = line.split('=', 1)[1]
        break

if not token:
    print("ERROR: No stored GitHub credentials found.")
    print("Please run: gh auth login  or  git credential fill")
    sys.exit(1)

headers = {
    'Authorization': 'token ' + token,
    'Accept': 'application/vnd.github.v3+json',
    'Content-Type': 'application/json'
}

# Get authenticated user
req = urllib.request.Request('https://api.github.com/user', headers=headers)
try:
    with urllib.request.urlopen(req) as r:
        user = json.loads(r.read())
        login = user['login']
        print("Authenticated as:", login)
except urllib.error.HTTPError as e:
    msg = json.loads(e.read()).get('message', '')
    print("Auth error:", e.code, msg)
    sys.exit(1)

# Check if repo already exists
repo_url = 'https://api.github.com/repos/' + login + '/IA_P1'
req2 = urllib.request.Request(repo_url, headers=headers)
repo_exists = False
try:
    with urllib.request.urlopen(req2) as r:
        repo = json.loads(r.read())
        repo_exists = True
        print("Repo already exists:", repo['html_url'], "| private:", repo['private'])
        # Make it public if private
        if repo['private']:
            patch_data = json.dumps({'private': False}).encode()
            req_patch = urllib.request.Request(repo_url, data=patch_data, headers=headers, method='PATCH')
            with urllib.request.urlopen(req_patch) as rp:
                updated = json.loads(rp.read())
                print("Made public:", updated['html_url'])
except urllib.error.HTTPError as e:
    if e.code == 404:
        print("Repo does not exist yet - creating...")
    else:
        print("Repo check error:", e.code)

# Create repo if needed
if not repo_exists:
    payload = json.dumps({
        'name': 'IA_P1',
        'private': False,
        'description': 'Classification Robuste - Detection de Fraudes Bancaires | Credit Card Fraud Detection | IA 2025-2026',
        'auto_init': False
    }).encode()
    req3 = urllib.request.Request(
        'https://api.github.com/user/repos',
        data=payload, headers=headers, method='POST'
    )
    try:
        with urllib.request.urlopen(req3) as r:
            repo = json.loads(r.read())
            print("Repo created:", repo['html_url'])
    except urllib.error.HTTPError as e:
        body = json.loads(e.read())
        print("Create error:", e.code, body.get('message', ''), body.get('errors', ''))
        sys.exit(1)

# Set remote and push
clone_url = 'https://github.com/' + login + '/IA_P1.git'
subprocess.run(['git', 'remote', 'remove', 'origin'], capture_output=True)
subprocess.run(['git', 'remote', 'add', 'origin', clone_url], check=True)
print("Remote set to:", clone_url)

# Push (git credential manager handles auth)
result = subprocess.run(['git', 'push', '-u', 'origin', 'main', '--force'],
    capture_output=True, text=True, encoding='utf-8', errors='replace')
if result.returncode == 0:
    print("PUSH SUCCESS!")
    print(result.stderr[:500] if result.stderr else "")
    print("Repo live at: https://github.com/" + login + "/IA_P1")
else:
    print("PUSH FAILED:")
    print(result.stderr[:500])
    sys.exit(1)
