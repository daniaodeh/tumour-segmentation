#!/usr/bin/env python3
import argparse
import io
import pickle
import tarfile
import zipfile
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

KNOWN_CHECKPOINT_KEYS = {
    "state_dict",
    "model_state_dict",
    "optimizer_state_dict",
    "checkpoint",
    "epoch",
    "step",
    "iteration",
    "global_step",
}


def is_tensor_like(obj):
    if torch is not None and isinstance(obj, torch.Tensor):
        return True
    name = type(obj).__name__
    return name in {"Tensor", "Parameter"}


def describe_obj(obj, max_items=5):
    if isinstance(obj, dict):
        keys = list(obj.keys())
        preview = keys[:max_items]
        return f"dict with {len(keys)} keys: {preview}"
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__} of length {len(obj)}"
    return f"{type(obj).__name__}"


def has_tensor_values(obj):
    if is_tensor_like(obj):
        return True
    if isinstance(obj, dict):
        return any(has_tensor_values(v) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return any(has_tensor_values(v) for v in obj)
    return False


def has_checkpoint_keys(obj):
    if isinstance(obj, dict):
        lower_keys = {k.lower() for k in obj.keys() if isinstance(k, str)}
        if lower_keys & KNOWN_CHECKPOINT_KEYS:
            return True
        return any(has_checkpoint_keys(v) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return any(has_checkpoint_keys(v) for v in obj)
    return False


def try_pickle_load(data_bytes):
    try:
        with io.BytesIO(data_bytes) as bio:
            unpickler = pickle.Unpickler(bio)
            visited_pid = []

            def persistent_load(pid):
                visited_pid.append(pid)
                return pid

            unpickler.persistent_load = persistent_load
            obj = unpickler.load()
        return obj, visited_pid, None
    except Exception as exc:
        return None, None, exc


def try_torch_load(data_bytes):
    if torch is None:
        return None, None
    try:
        with io.BytesIO(data_bytes) as bio:
            obj = torch.load(bio, map_location="cpu")
        return obj, None
    except Exception as exc:
        return None, exc


def inspect_tar_file(tar_path: Path, member_name: str):
    with tarfile.open(tar_path, mode="r:*") as tar:
        members = tar.getmembers()
        print(f"Tar archive: {tar_path}")
        print(f"Contained {len(members)} entries")
        for m in members:
            print(f"  - {m.name} ({m.size} bytes)")

        candidate = None
        candidates = [member_name, member_name.replace('.pck', '.pkl'), member_name.replace('.pkl', '.pck')]
        lower_members = {m.name.lower(): m for m in members}
        for name in candidates:
            if name in lower_members:
                candidate = lower_members[name]
                break
        if candidate is None:
            for name in candidates:
                for member_name_lower, member in lower_members.items():
                    if member_name_lower.endswith(name.lower()):
                        candidate = member
                        break
                if candidate is not None:
                    break

        if candidate is None:
            print(f"\nCould not find {member_name} in archive.")
            print("Search candidates:")
            lower_names = [m.name.lower() for m in members]
            for pattern in ["data.pkl", "data.pck", "*.pkl", "*.pck", "checkpoint", "state"]:
                hits = [m.name for m in members if pattern.strip('*') in m.name.lower()]
                if hits:
                    print(f"  {pattern}: {hits[:5]}")
            return

        print(f"\nInspecting archive member: {candidate.name} ({candidate.size} bytes)")
        fobj = tar.extractfile(candidate)
        if fobj is None:
            print("Failed to open the archive member.")
            return

        analyze_member(candidate.name, fobj.read())


def inspect_zip_file(tar_path: Path, member_name: str):
    with zipfile.ZipFile(tar_path, mode="r") as zipf:
        members = zipf.infolist()
        print(f"Zip archive: {tar_path}")
        print(f"Contained {len(members)} entries")
        for m in members:
            print(f"  - {m.filename} ({m.file_size} bytes)")

        candidate = None
        candidates = [member_name, member_name.replace('.pck', '.pkl'), member_name.replace('.pkl', '.pck')]
        lower_members = {m.filename.lower(): m for m in members}
        for name in candidates:
            if name in lower_members:
                candidate = lower_members[name]
                break
        if candidate is None:
            for name in candidates:
                for member_name_lower, member in lower_members.items():
                    if member_name_lower.endswith(name.lower()):
                        candidate = member
                        break
                if candidate is not None:
                    break

        if candidate is None:
            print(f"\nCould not find {member_name} in archive.")
            print("Search candidates:")
            lower_names = [m.filename.lower() for m in members]
            for pattern in ["data.pkl", "data.pck", "*.pkl", "*.pck", "checkpoint", "state"]:
                hits = [m.filename for m in members if pattern.strip('*') in m.filename.lower()]
                if hits:
                    print(f"  {pattern}: {hits[:5]}")
            return

        print(f"\nInspecting archive member: {candidate.filename} ({candidate.file_size} bytes)")
        with zipf.open(candidate, 'r') as fobj:
            analyze_member(candidate.filename, fobj.read())


def analyze_member(member_name: str, data_bytes: bytes):
    print(f"Read {len(data_bytes)} bytes from {member_name}\n")

    obj, pids, err = try_pickle_load(data_bytes)
    if obj is not None:
        print("Loaded with pickle.Unpickler")
        print(f"Type: {type(obj).__name__}")
        print(f"Description: {describe_obj(obj)}")
        if pids:
            print(f"Persistent IDs encountered: {pids[:10]}")
    else:
        print(f"Pickle load failed: {err}")

    if torch is not None:
        torch_obj, torch_err = try_torch_load(data_bytes)
        if torch_obj is not None:
            print("\nLoaded with torch.load")
            print(f"Type: {type(torch_obj).__name__}")
            print(f"Description: {describe_obj(torch_obj)}")
            obj = torch_obj
        else:
            print(f"torch.load failed: {torch_err}")
    else:
        print("torch not installed, skipping torch.load")

    if obj is not None:
        checkpoint_like = has_checkpoint_keys(obj)
        tensor_like = has_tensor_values(obj)
        print(f"\nCheckpoint-like: {checkpoint_like}")
        print(f"Weights-like (tensor values): {tensor_like}")
        if checkpoint_like:
            print("This object appears to contain checkpoint metadata or a state dict.")
        elif tensor_like:
            print("This object appears to contain raw weights / tensors.")
        else:
            print("Could not confidently classify the object as checkpoint or weights.")


def inspect_file(tar_path: Path, member_name: str):
    try:
        inspect_tar_file(tar_path, member_name)
    except tarfile.ReadError as tar_err:
        print(f"Tar open failed: {tar_err}. Trying zip archive instead.")
        inspect_zip_file(tar_path, member_name)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a tar archive for data.pkl/data.pck and detect checkpoint/weight content."
    )
    parser.add_argument("tar_path", help="Path to model.tar archive")
    parser.add_argument(
        "--member",
        default="data.pkl",
        help="Name of the member inside the archive to inspect (default: data.pkl)",
    )
    args = parser.parse_args()

    tar_path = Path(args.tar_path)
    if not tar_path.exists():
        raise FileNotFoundError(f"Archive not found: {tar_path}")

    inspect_file(tar_path, args.member)


if __name__ == "__main__":
    main()
