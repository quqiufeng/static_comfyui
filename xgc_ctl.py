#!/usr/bin/env python3
"""Xiangongyun instance API controller."""
import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error

API_BASE = "https://api.xiangongyun.com"


def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                if os.environ.get(k) is None:
                    os.environ[k] = v


_load_env()


def _env(key):
    v = os.environ.get(key)
    if not v:
        raise SystemExit(f"missing env var {key}")
    return v


def _token():
    return _env("XGC_API_TOKEN")


def _request(path, method="GET", body=None):
    url = API_BASE + path
    headers = {
        "Authorization": f"Bearer {_token()}",
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
        raise SystemExit(1)


def cmd_list(_):
    r = _request("/open/instances")
    print(json.dumps(r, indent=2, ensure_ascii=False))


def cmd_deploy(args):
    body = {
        "gpu_model": args.gpu_model,
        "gpu_count": args.gpu_count,
        "data_center_id": args.data_center_id,
        "image": args.image,
        "image_type": args.image_type,
    }
    if args.name:
        body["name"] = args.name
    r = _request("/open/instance/deploy", method="POST", body=body)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    if r.get("success") and r.get("data", {}).get("id"):
        print(f"instance_id={r['data']['id']}")


def _find_instance(instance_id):
    r = _request("/open/instances")
    inst = next((i for i in r.get("data", {}).get("list", []) if i["id"] == instance_id), None)
    if inst is None:
        print(f"instance {instance_id} not found", file=sys.stderr)
        raise SystemExit(1)
    return inst


def cmd_info(args):
    inst = _find_instance(args.id)
    print(f"id:           {inst['id']}")
    print(f"name:         {inst.get('name', '')}")
    print(f"status:       {inst.get('status', 'unknown')}")
    print(f"gpu:          {inst.get('gpu_model', '')} x {inst.get('gpu_used', 0)}")
    print(f"cpu:          {inst.get('cpu_core_count', 0)} cores")
    print(f"memory:       {inst.get('memory_size', 0) / (1024**3):.1f} GB")
    print(f"price/hour:   {inst.get('price_per_hour', 0)} CNY")
    print(f"ssh_domain:   {inst.get('ssh_domain', '')}")
    print(f"ssh_port:     {inst.get('ssh_port', '')}")
    print(f"ssh_user:     {inst.get('ssh_user', 'root')}")
    print(f"password:     {inst.get('password', '')}")
    print(f"jupyter_url:  {inst.get('jupyter_url', '')}")
    print(f"web_url:      {inst.get('web_url', '')}")


def cmd_status(args):
    inst = _find_instance(args.id)
    print(json.dumps(inst, indent=2, ensure_ascii=False))


def cmd_wait(args):
    while True:
        r = _request("/open/instances")
        inst = next((i for i in r.get("data", {}).get("list", []) if i["id"] == args.id), None)
        if inst is None:
            print(f"instance {args.id} not found", file=sys.stderr)
            raise SystemExit(1)
        status = inst.get("status", "unknown")
        print(f"{args.id}: {status}")
        if status == args.status:
            print(json.dumps(inst, indent=2, ensure_ascii=False))
            return
        time.sleep(args.interval)


def cmd_shutdown(args):
    r = _request("/open/instance/shutdown_release_gpu", method="POST", body={"id": args.id})
    print(json.dumps(r, indent=2, ensure_ascii=False))


def cmd_destroy(args):
    r = _request("/open/instance/destroy", method="POST", body={"id": args.id})
    print(json.dumps(r, indent=2, ensure_ascii=False))


def cmd_shutdown_destroy(args):
    r = _request("/open/instance/shutdown_destroy", method="POST", body={"id": args.id})
    print(json.dumps(r, indent=2, ensure_ascii=False))


def cmd_ssh(args):
    r = _request("/open/instances")
    inst = next((i for i in r.get("data", {}).get("list", []) if i["id"] == args.id), None)
    if inst is None:
        print(f"instance {args.id} not found", file=sys.stderr)
        raise SystemExit(1)
    domain = inst.get("ssh_domain", "")
    port = inst.get("ssh_port", "")
    user = inst.get("ssh_user", "root")
    password = inst.get("password", "")
    if not domain or not port:
        print("SSH info not available yet", file=sys.stderr)
        raise SystemExit(1)
    cmd = f"sshpass -p '{password}' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {port} {user}@{domain}"
    print(cmd)


def main():
    p = argparse.ArgumentParser(description="Xiangongyun instance API controller")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list instances")

    deploy = sub.add_parser("deploy", help="deploy a new instance")
    deploy.add_argument("--gpu-model", default="NVIDIA GeForce RTX 4090")
    deploy.add_argument("--gpu-count", type=int, default=1)
    deploy.add_argument("--data-center-id", type=int, default=1)
    deploy.add_argument("--image", default=os.environ.get("XGC_IMAGE_ID", ""))
    deploy.add_argument("--image-type", default="private")
    deploy.add_argument("--name", default="")

    status = sub.add_parser("status", help="show instance status (raw json)")
    status.add_argument("id", help="instance id")

    info = sub.add_parser("info", help="show key access info for an instance")
    info.add_argument("id", help="instance id")

    wait = sub.add_parser("wait", help="wait for instance status")
    wait.add_argument("id", help="instance id")
    wait.add_argument("--status", default="running", help="target status")
    wait.add_argument("--interval", type=int, default=10, help="poll interval seconds")

    shutdown = sub.add_parser("shutdown", help="shutdown and release GPU")
    shutdown.add_argument("id", help="instance id")

    destroy = sub.add_parser("destroy", help="destroy instance")
    destroy.add_argument("id", help="instance id")

    shutdown_destroy = sub.add_parser("shutdown_destroy", help="shutdown and destroy instance (stops billing)")
    shutdown_destroy.add_argument("id", help="instance id")

    ssh = sub.add_parser("ssh", help="print ssh command")
    ssh.add_argument("id", help="instance id")

    args = p.parse_args()
    globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    main()
