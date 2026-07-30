#!/usr/bin/env python3
"""Create a read-only, checksum-bound inventory of G1 pipeline evidence.

The command never copies, removes, or edits an inventoried artifact.  It writes
four new index files below ``--output`` and excludes that whole output tree
from traversal, even when the output directory is nested below an input root.

``--hash-mode metadata`` is intended for quick audits: it hashes regular files
whose names or extensions normally contain manifests, configuration, source,
logs, or tabular metadata.  ``all`` (the default) hashes every regular file,
while ``none`` records only filesystem metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import stat
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


SCHEMA_VERSION = "daaam.g1_semantic_pipeline_artifact_inventory.v1"
PROVENANCE_SCHEMA_VERSION = (
    "daaam.g1_semantic_pipeline_artifact_inventory_provenance.v1"
)
SUMMARY_SCHEMA_VERSION = "daaam.g1_semantic_pipeline_artifact_inventory_summary.v1"
HASH_CHUNK_BYTES = 4 * 1024 * 1024
OUTPUT_FILENAMES = (
    "artifact_inventory.jsonl",
    "artifact_inventory.csv",
    "inventory_summary.json",
    "inventory_provenance.json",
)

# In metadata mode, hash content that defines lineage, configuration, execution,
# or compact numerical/tabular reports.  Large binary sensor products are still
# fully stat'ed but are intentionally not read.
METADATA_EXTENSIONS = frozenset(
    {
        ".bash",
        ".cfg",
        ".cmake",
        ".conf",
        ".csv",
        ".env",
        ".fish",
        ".h",
        ".hpp",
        ".htm",
        ".html",
        ".ini",
        ".json",
        ".jsonl",
        ".lock",
        ".log",
        ".md",
        ".py",
        ".rst",
        ".sh",
        ".toml",
        ".tsv",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
        ".zsh",
    }
)
METADATA_FILENAMES = frozenset(
    {
        "cmakelists.txt",
        "dockerfile",
        "license",
        "makefile",
        "manifest",
        "notice",
        "pipeline",
        "readme",
    }
)

CSV_FIELDS = (
    "schema",
    "artifact_id",
    "stage",
    "root",
    "root_path",
    "relative_path",
    "absolute_path",
    "file_kind",
    "extension",
    "size_bytes",
    "mtime_ns",
    "ctime_ns",
    "sha256",
    "hash_status",
    "bytes_hashed",
    "symlink_target",
    "symlink_target_absolute",
    "target_file_kind",
    "target_size_bytes",
    "target_mtime_ns",
    "mode_octal",
    "mode_string",
    "uid",
    "gid",
    "device",
    "inode",
    "nlink",
    "traversal_status",
    "error",
    "error_type",
)


@dataclass(frozen=True)
class RootSpec:
    """A caller-assigned logical root and its non-symlink-resolved path."""

    name: str
    path: Path
    argument: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def absolute_without_resolving(path: Path) -> Path:
    """Return an absolute lexical path while preserving final symlink identity."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def parse_root(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--root must use NAME=PATH syntax, got {value!r}"
        )
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("root NAME must not be empty")
    if not raw_path:
        raise argparse.ArgumentTypeError(f"root PATH must not be empty for {name!r}")
    if any(character in name for character in ("\x00", "\n", "\r")):
        raise argparse.ArgumentTypeError("root NAME must not contain control characters")
    return name, raw_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help=(
            "Logical evidence root. Repeat for every upstream dataset or stage "
            "directory that must be inventoried."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory in which the four inventory products will be atomically written.",
    )
    parser.add_argument(
        "--hash-mode",
        choices=("all", "metadata", "none"),
        default="all",
        help=(
            "all: hash every regular file; metadata: hash manifests/config/source/"
            "logs/tables only; none: do not read file content (default: all)."
        ),
    )
    parser.add_argument(
        "--follow-symlinks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Follow symlink targets for hashing and directory traversal. Symlink "
            "objects and their literal targets are always recorded (default: false)."
        ),
    )
    return parser.parse_args(argv)


def build_root_specs(values: Sequence[str]) -> list[RootSpec]:
    roots: list[RootSpec] = []
    used_names: set[str] = set()
    for value in values:
        name, raw_path = parse_root(value)
        if name in used_names:
            raise ValueError(f"Duplicate --root NAME is not allowed: {name!r}")
        used_names.add(name)
        roots.append(
            RootSpec(
                name=name,
                path=absolute_without_resolving(Path(raw_path)),
                argument=value,
            )
        )
    return roots


def is_path_within(path: Path, directory: Path) -> bool:
    try:
        os.path.commonpath((os.fspath(path), os.fspath(directory)))
    except ValueError:
        return False
    return os.path.commonpath((os.fspath(path), os.fspath(directory))) == os.fspath(
        directory
    )


def classify_mode(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISCHR(mode):
        return "character_device"
    if stat.S_ISBLK(mode):
        return "block_device"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    return "other"


def metadata_extension(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix else "<none>"


def should_hash_metadata(path: Path) -> bool:
    lower_name = path.name.lower()
    if path.suffix.lower() in METADATA_EXTENSIONS:
        return True
    if lower_name in METADATA_FILENAMES:
        return True
    return any(
        lower_name.startswith(prefix + ".") for prefix in METADATA_FILENAMES
    )


def sanitize_error_message(error: BaseException) -> str:
    return str(error).replace("\r", "\\r").replace("\n", "\\n")


def add_error(record: dict[str, Any], operation: str, error: BaseException) -> None:
    record["_errors"].append(
        {
            "type": f"{operation}:{type(error).__name__}",
            "message": (
                f"{operation}: {type(error).__name__}: "
                f"{sanitize_error_message(error)}"
            ),
        }
    )


def add_consistency_error(
    record: dict[str, Any], operation: str, message: str
) -> None:
    record["_errors"].append(
        {
            "type": f"{operation}:ArtifactChanged",
            "message": f"{operation}: ArtifactChanged: {message}",
        }
    )


def same_file_snapshot(first: os.stat_result, second: os.stat_result) -> bool:
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    return all(getattr(first, field) == getattr(second, field) for field in fields)


def stream_sha256(
    path: Path, expected: os.stat_result
) -> tuple[str | None, int, BaseException | str | None]:
    """Hash one file without loading it into memory and detect common races."""

    digest = hashlib.sha256()
    bytes_read = 0
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not same_file_snapshot(expected, opened):
                return (
                    None,
                    0,
                    "file identity or metadata changed before hashing began",
                )
            while True:
                chunk = handle.read(HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                bytes_read += len(chunk)
            after_read = os.fstat(handle.fileno())
        after_path = path.stat()
        if not same_file_snapshot(opened, after_read):
            return None, bytes_read, "open file changed while it was being hashed"
        if not same_file_snapshot(after_read, after_path):
            return None, bytes_read, "path target changed while it was being hashed"
        return digest.hexdigest(), bytes_read, None
    except (OSError, ValueError) as error:
        return None, bytes_read, error


def artifact_id(root_name: str, relative_path: str) -> str:
    payload = f"{root_name}\0{relative_path}".encode(
        "utf-8", errors="surrogateescape"
    )
    return hashlib.sha256(payload).hexdigest()


def base_record(
    root: RootSpec, path: Path, relative_parts: tuple[str, ...]
) -> dict[str, Any]:
    relative_path = "." if not relative_parts else "/".join(relative_parts)
    return {
        "schema": SCHEMA_VERSION,
        "artifact_id": artifact_id(root.name, relative_path),
        "stage": root.name,
        "root": root.name,
        "root_path": os.fspath(root.path),
        "relative_path": relative_path,
        "absolute_path": os.fspath(path),
        "file_kind": "missing",
        "extension": None,
        "size_bytes": None,
        "mtime_ns": None,
        "ctime_ns": None,
        "sha256": None,
        "hash_status": "not_applicable",
        "bytes_hashed": 0,
        "symlink_target": None,
        "symlink_target_absolute": None,
        "target_file_kind": None,
        "target_size_bytes": None,
        "target_mtime_ns": None,
        "mode_octal": None,
        "mode_string": None,
        "uid": None,
        "gid": None,
        "device": None,
        "inode": None,
        "nlink": None,
        "traversal_status": "not_traversed",
        "error": None,
        "error_type": None,
        "_errors": [],
    }


def apply_stat(record: dict[str, Any], metadata: os.stat_result) -> None:
    record.update(
        {
            "file_kind": classify_mode(metadata.st_mode),
            "size_bytes": int(metadata.st_size),
            "mtime_ns": int(metadata.st_mtime_ns),
            "ctime_ns": int(metadata.st_ctime_ns),
            "mode_octal": f"{stat.S_IMODE(metadata.st_mode):04o}",
            "mode_string": stat.filemode(metadata.st_mode),
            "uid": int(metadata.st_uid),
            "gid": int(metadata.st_gid),
            "device": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
            "nlink": int(metadata.st_nlink),
        }
    )


def finalize_record(
    record: dict[str, Any], relative_parts: tuple[str, ...]
) -> dict[str, Any]:
    kind = record["file_kind"]
    target_kind = record["target_file_kind"]
    if relative_parts:
        if len(relative_parts) > 1 or kind == "directory" or target_kind == "directory":
            record["stage"] = relative_parts[0]
    errors = record.pop("_errors")
    if errors:
        record["error"] = "; ".join(item["message"] for item in errors)
        record["error_type"] = "|".join(item["type"] for item in errors)
    return record


class EvidenceWalker:
    """Filesystem walker that records failures instead of hiding them."""

    def __init__(
        self,
        *,
        output: Path,
        hash_mode: str,
        follow_symlinks: bool,
    ) -> None:
        self.output = output
        self.output_real = Path(os.path.realpath(output))
        self.hash_mode = hash_mode
        self.follow_symlinks = follow_symlinks
        self.visited_directories: set[tuple[int, int]] = set()

    def excluded(self, path: Path) -> bool:
        lexical = absolute_without_resolving(path)
        if is_path_within(lexical, self.output):
            return True
        # This prevents a followed symlink from re-entering the live output tree.
        real = Path(os.path.realpath(path))
        return is_path_within(real, self.output_real)

    def hash_regular_file(
        self,
        record: dict[str, Any],
        path: Path,
        metadata: os.stat_result,
    ) -> None:
        if self.hash_mode == "none":
            record["hash_status"] = "skipped_hash_mode_none"
            return
        if self.hash_mode == "metadata" and not should_hash_metadata(path):
            record["hash_status"] = "skipped_non_metadata"
            return
        digest, bytes_read, error = stream_sha256(path, metadata)
        record["bytes_hashed"] = bytes_read
        if error is None:
            record["sha256"] = digest
            record["hash_status"] = "computed"
        elif isinstance(error, BaseException):
            record["hash_status"] = "error"
            add_error(record, "sha256", error)
        else:
            record["hash_status"] = "error"
            add_consistency_error(record, "sha256", error)

    def list_directory(
        self, record: dict[str, Any], path: Path, metadata: os.stat_result
    ) -> list[os.DirEntry[str]] | None:
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if identity in self.visited_directories:
            record["traversal_status"] = "skipped_already_visited_directory"
            return None
        self.visited_directories.add(identity)
        try:
            with os.scandir(path) as scan:
                entries = list(scan)
        except OSError as error:
            record["traversal_status"] = "scan_error"
            add_error(record, "scandir", error)
            return None
        entries.sort(key=lambda item: os.fsencode(item.name))
        record["traversal_status"] = "traversed"
        return entries

    def walk(self, root: RootSpec) -> Iterator[dict[str, Any]]:
        # De-duplicate/cycle-protect within one logical root, but inventory an
        # overlapping path again when the caller intentionally assigns it a
        # second root identity.
        self.visited_directories.clear()
        yield from self._walk_path(root, root.path, ())

    def _walk_path(
        self,
        root: RootSpec,
        path: Path,
        relative_parts: tuple[str, ...],
    ) -> Iterator[dict[str, Any]]:
        if self.excluded(path):
            return
        record = base_record(root, path, relative_parts)
        try:
            lstat_result = path.lstat()
        except OSError as error:
            add_error(record, "lstat", error)
            yield finalize_record(record, relative_parts)
            return

        apply_stat(record, lstat_result)
        kind = record["file_kind"]
        record["extension"] = (
            metadata_extension(path) if kind in {"file", "symlink"} else None
        )

        entries: list[os.DirEntry[str]] | None = None
        if kind == "file":
            self.hash_regular_file(record, path, lstat_result)
            record["traversal_status"] = "not_a_directory"
        elif kind == "directory":
            entries = self.list_directory(record, path, lstat_result)
        elif kind == "symlink":
            try:
                target = os.readlink(path)
                record["symlink_target"] = target
                if os.path.isabs(target):
                    target_absolute = os.path.normpath(target)
                else:
                    target_absolute = os.path.normpath(
                        os.path.join(os.fspath(path.parent), target)
                    )
                record["symlink_target_absolute"] = target_absolute
            except OSError as error:
                add_error(record, "readlink", error)

            if not self.follow_symlinks:
                record["traversal_status"] = "skipped_symlink_policy"
            else:
                try:
                    target_stat = path.stat()
                except OSError as error:
                    record["traversal_status"] = "broken_or_unreadable_symlink"
                    add_error(record, "stat_symlink_target", error)
                else:
                    target_kind = classify_mode(target_stat.st_mode)
                    record["target_file_kind"] = target_kind
                    record["target_size_bytes"] = int(target_stat.st_size)
                    record["target_mtime_ns"] = int(target_stat.st_mtime_ns)
                    if target_kind == "file":
                        self.hash_regular_file(record, path, target_stat)
                        record["traversal_status"] = "followed_file_symlink"
                    elif target_kind == "directory":
                        entries = self.list_directory(record, path, target_stat)
                    else:
                        record["traversal_status"] = (
                            "followed_symlink_to_non_regular_target"
                        )
        else:
            record["traversal_status"] = "unsupported_special_file"

        yield finalize_record(record, relative_parts)
        if entries:
            for entry in entries:
                child_path = path / entry.name
                if self.excluded(child_path):
                    continue
                yield from self._walk_path(
                    root,
                    child_path,
                    relative_parts + (entry.name,),
                )


def empty_bucket() -> dict[str, int]:
    return {
        "records": 0,
        "files": 0,
        "directories": 0,
        "symlinks": 0,
        "other": 0,
        "total_bytes": 0,
        "hashed_files": 0,
        "bytes_hashed": 0,
        "failures": 0,
    }


def update_bucket(bucket: dict[str, int], record: dict[str, Any]) -> None:
    bucket["records"] += 1
    kind = record["file_kind"]
    if kind == "file":
        bucket["files"] += 1
        bucket["total_bytes"] += int(record["size_bytes"] or 0)
    elif kind == "directory":
        bucket["directories"] += 1
    elif kind == "symlink":
        bucket["symlinks"] += 1
    else:
        bucket["other"] += 1
    if record["hash_status"] == "computed":
        bucket["hashed_files"] += 1
    bucket["bytes_hashed"] += int(record["bytes_hashed"] or 0)
    if record["error"] is not None:
        bucket["failures"] += 1


class InventorySummary:
    def __init__(
        self,
        *,
        roots: Sequence[RootSpec],
        hash_mode: str,
        follow_symlinks: bool,
        started_at: str,
    ) -> None:
        self.roots = roots
        self.hash_mode = hash_mode
        self.follow_symlinks = follow_symlinks
        self.started_at = started_at
        self.totals = empty_bucket()
        self.by_root = {root.name: empty_bucket() for root in roots}
        self.by_stage: dict[str, dict[str, int]] = {}
        self.by_extension: dict[str, dict[str, int]] = {}
        self.by_kind: Counter[str] = Counter()
        self.failures_by_error_type: Counter[str] = Counter()

    def add(self, record: dict[str, Any]) -> None:
        update_bucket(self.totals, record)
        update_bucket(self.by_root[record["root"]], record)
        stage_bucket = self.by_stage.setdefault(record["stage"], empty_bucket())
        update_bucket(stage_bucket, record)
        self.by_kind[record["file_kind"]] += 1
        if record["file_kind"] == "file":
            extension = record["extension"] or "<none>"
            extension_bucket = self.by_extension.setdefault(
                extension,
                {
                    "files": 0,
                    "total_bytes": 0,
                    "hashed_files": 0,
                    "failures": 0,
                },
            )
            extension_bucket["files"] += 1
            extension_bucket["total_bytes"] += int(record["size_bytes"] or 0)
            if record["hash_status"] == "computed":
                extension_bucket["hashed_files"] += 1
            if record["error"] is not None:
                extension_bucket["failures"] += 1
        if record["error_type"]:
            for error_type in record["error_type"].split("|"):
                self.failures_by_error_type[error_type] += 1

    def as_dict(self, finished_at: str) -> dict[str, Any]:
        return {
            "schema": SUMMARY_SCHEMA_VERSION,
            "inventory_schema": SCHEMA_VERSION,
            "started_at_utc": self.started_at,
            "finished_at_utc": finished_at,
            "hash_mode": self.hash_mode,
            "follow_symlinks": self.follow_symlinks,
            "byte_accounting": (
                "total_bytes counts lstat sizes of regular files only; symlink "
                "inode sizes and followed-target duplicates are excluded"
            ),
            "roots": [
                {"name": root.name, "path": os.fspath(root.path)}
                for root in self.roots
            ],
            "totals": self.totals,
            "by_root": dict(sorted(self.by_root.items())),
            "by_stage": dict(sorted(self.by_stage.items())),
            "by_extension": dict(sorted(self.by_extension.items())),
            "by_file_kind": dict(sorted(self.by_kind.items())),
            "failures_by_error_type": dict(
                sorted(self.failures_by_error_type.items())
            ),
        }


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)
    return value


def fsync_text_file(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def create_text_temp(output: Path, final_name: str) -> tuple[Path, Any]:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        errors="backslashreplace",
        newline="",
        prefix=f".{final_name}.",
        suffix=".tmp",
        dir=output,
        delete=False,
    )
    return Path(handle.name), handle


def create_json_temp(output: Path, final_name: str, payload: Any) -> Path:
    temporary, handle = create_text_temp(output, final_name)
    try:
        json.dump(
            payload,
            handle,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
        fsync_text_file(handle)
    except BaseException:
        handle.close()
        cleanup_temporaries((temporary,))
        raise
    finally:
        if not handle.closed:
            handle.close()
    return temporary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def install_outputs(output: Path, temporary_files: dict[str, Path]) -> None:
    # The provenance marker is installed last: its presence therefore denotes a
    # completed four-file publication rather than a still-running inventory.
    order = (
        "artifact_inventory.jsonl",
        "artifact_inventory.csv",
        "inventory_summary.json",
        "inventory_provenance.json",
    )
    for name in order:
        os.replace(temporary_files[name], output / name)
    fsync_directory(output)


def root_snapshot(root: RootSpec) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": root.name,
        "argument": root.argument,
        "path": os.fspath(root.path),
        "exists": False,
        "file_kind": "missing",
        "size_bytes": None,
        "mtime_ns": None,
        "error": None,
    }
    try:
        metadata = root.path.lstat()
    except OSError as error:
        result["error"] = (
            f"lstat: {type(error).__name__}: {sanitize_error_message(error)}"
        )
    else:
        result.update(
            {
                "exists": True,
                "file_kind": classify_mode(metadata.st_mode),
                "size_bytes": int(metadata.st_size),
                "mtime_ns": int(metadata.st_mtime_ns),
            }
        )
    return result


def build_provenance(
    *,
    args: argparse.Namespace,
    roots: Sequence[RootSpec],
    output: Path,
    started_at: str,
    finished_at: str,
    summary: dict[str, Any],
    temporary_files: dict[str, Path],
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    products: dict[str, dict[str, Any]] = {}
    for name in (
        "artifact_inventory.jsonl",
        "artifact_inventory.csv",
        "inventory_summary.json",
    ):
        path = temporary_files[name]
        products[name] = {
            "path": os.fspath(output / name),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema": PROVENANCE_SCHEMA_VERSION,
        "inventory_schema": SCHEMA_VERSION,
        "summary_schema": SUMMARY_SCHEMA_VERSION,
        "status": "complete",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "command": [os.fspath(script_path), *sys.argv[1:]],
        "argv": list(sys.argv),
        "working_directory": os.getcwd(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "executable": sys.executable,
            "uid": os.getuid() if hasattr(os, "getuid") else None,
            "gid": os.getgid() if hasattr(os, "getgid") else None,
        },
        "implementation": {
            "script_path": os.fspath(script_path),
            "script_sha256": sha256_file(script_path),
            "hash_algorithm": "sha256",
            "hash_chunk_bytes": HASH_CHUNK_BYTES,
        },
        "configuration": {
            "hash_mode": args.hash_mode,
            "follow_symlinks": args.follow_symlinks,
            "metadata_extensions": sorted(METADATA_EXTENSIONS),
            "metadata_filenames": sorted(METADATA_FILENAMES),
            "stage_derivation": (
                "first root-relative directory component; root NAME for the root "
                "itself and root-level non-directory artifacts"
            ),
            "output_path": os.fspath(output),
            "output_tree_excluded": True,
        },
        "roots_at_completion": [root_snapshot(root) for root in roots],
        "products": products,
        "inventory_totals": summary["totals"],
        "integrity_note": (
            "Product hashes bind JSONL, CSV, and summary. The provenance file is "
            "not self-hashed; it is atomically installed last as completion marker."
        ),
    }


def cleanup_temporaries(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # Preserve the original exception from inventory generation.
            pass


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots = build_root_specs(args.root)
    output_requested = absolute_without_resolving(args.output)
    if output_requested.is_symlink():
        raise ValueError(
            f"Refusing a symlink as --output directory: {output_requested}"
        )
    output = output_requested.resolve(strict=False)
    for root in roots:
        if is_path_within(root.path, output):
            raise ValueError(
                f"Evidence root {root.name!r} lies inside --output and would be "
                f"entirely excluded: {root.path}"
            )

    output.mkdir(parents=True, exist_ok=True)
    if not output.is_dir():
        raise NotADirectoryError(f"--output is not a directory: {output}")

    started_at = utc_now()
    temporary_files: dict[str, Path] = {}
    open_handles: list[Any] = []
    try:
        jsonl_temp, jsonl_handle = create_text_temp(
            output, "artifact_inventory.jsonl"
        )
        temporary_files["artifact_inventory.jsonl"] = jsonl_temp
        open_handles.append(jsonl_handle)
        csv_temp, csv_handle = create_text_temp(output, "artifact_inventory.csv")
        temporary_files["artifact_inventory.csv"] = csv_temp
        open_handles.append(csv_handle)

        csv_writer = csv.DictWriter(
            csv_handle,
            fieldnames=CSV_FIELDS,
            extrasaction="raise",
            lineterminator="\n",
        )
        csv_writer.writeheader()
        summary_builder = InventorySummary(
            roots=roots,
            hash_mode=args.hash_mode,
            follow_symlinks=args.follow_symlinks,
            started_at=started_at,
        )
        walker = EvidenceWalker(
            output=output,
            hash_mode=args.hash_mode,
            follow_symlinks=args.follow_symlinks,
        )
        for root in roots:
            for record in walker.walk(root):
                jsonl_handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=True,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                csv_writer.writerow(
                    {field: csv_value(record[field]) for field in CSV_FIELDS}
                )
                summary_builder.add(record)

        fsync_text_file(jsonl_handle)
        fsync_text_file(csv_handle)
        jsonl_handle.close()
        csv_handle.close()
        open_handles.clear()

        finished_at = utc_now()
        summary = summary_builder.as_dict(finished_at)
        temporary_files["inventory_summary.json"] = create_json_temp(
            output, "inventory_summary.json", summary
        )
        provenance = build_provenance(
            args=args,
            roots=roots,
            output=output,
            started_at=started_at,
            finished_at=finished_at,
            summary=summary,
            temporary_files=temporary_files,
        )
        temporary_files["inventory_provenance.json"] = create_json_temp(
            output, "inventory_provenance.json", provenance
        )
        install_outputs(output, temporary_files)
        temporary_files.clear()
    except BaseException:
        for handle in open_handles:
            try:
                handle.close()
            except OSError:
                pass
        cleanup_temporaries(list(temporary_files.values()))
        raise

    print(
        json.dumps(
            {
                "status": "complete",
                "output": os.fspath(output),
                "hash_mode": args.hash_mode,
                "follow_symlinks": args.follow_symlinks,
                "records": summary["totals"]["records"],
                "files": summary["totals"]["files"],
                "total_bytes": summary["totals"]["total_bytes"],
                "hashed_files": summary["totals"]["hashed_files"],
                "failures": summary["totals"]["failures"],
                "products": [
                    os.fspath(output / filename) for filename in OUTPUT_FILENAMES
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
