#!/usr/bin/env bash
set -euo pipefail

root=/mnt/daiyang/vla
key="$root/.secret/migration_ed25519_8_130_54_48"
remote=root@8.130.54.48
destination=/mnt/data_nas/deeprobotics/daiyang
ssh_options=(
  -i "$key" -o IdentitiesOnly=yes -o BatchMode=yes
  -o ConnectTimeout=15 -o StrictHostKeyChecking=yes
)

probe_src=$(mktemp -d "$root/artifacts/migration/.probe_src.XXXXXX")
probe_dst=$(ssh "${ssh_options[@]}" "$remote" \
  "mktemp -d '$destination/.probe_dst.XXXXXX'")
case "$probe_src" in
  "$root"/artifacts/migration/.probe_src.*) ;;
  *) exit 2 ;;
esac
case "$probe_dst" in
  "$destination"/.probe_dst.*) ;;
  *) exit 2 ;;
esac

cleanup_local() {
  unlink "$probe_src/payload/symlink.bin" 2>/dev/null || true
  unlink "$probe_src/payload/hardlink.bin" 2>/dev/null || true
  unlink "$probe_src/payload/regular.bin" 2>/dev/null || true
  unlink "$probe_src/payload/sparse.bin" 2>/dev/null || true
  rmdir "$probe_src/payload" 2>/dev/null || true
  rmdir "$probe_src" 2>/dev/null || true
}
cleanup_remote() {
  ssh "${ssh_options[@]}" "$remote" "
    unlink '$probe_dst/payload/symlink.bin' 2>/dev/null || true
    unlink '$probe_dst/payload/hardlink.bin' 2>/dev/null || true
    unlink '$probe_dst/payload/regular.bin' 2>/dev/null || true
    unlink '$probe_dst/payload/sparse.bin' 2>/dev/null || true
    rmdir '$probe_dst/payload' 2>/dev/null || true
    rmdir '$probe_dst' 2>/dev/null || true" || true
}
cleanup() {
  cleanup_local
  cleanup_remote
}
trap cleanup EXIT

mkdir -m 750 "$probe_src/payload"
dd if=/dev/urandom of="$probe_src/payload/regular.bin" \
  bs=4096 count=4 status=none
chmod 640 "$probe_src/payload/regular.bin"
chown 12345:23456 "$probe_src/payload/regular.bin"
touch -t 202608250101.02 "$probe_src/payload/regular.bin"
ln "$probe_src/payload/regular.bin" "$probe_src/payload/hardlink.bin"
ln -s regular.bin "$probe_src/payload/symlink.bin"
truncate -s 10485760 "$probe_src/payload/sparse.bin"
dd if=/dev/urandom of="$probe_src/payload/sparse.bin" \
  bs=4096 count=1 seek=128 conv=notrunc status=none
expected_sha=$(sha256sum "$probe_src/payload/regular.bin" | awk '{print $1}')
expected_mtime=$(stat -c '%Y' "$probe_src/payload/regular.bin")
rsync -aHSx --numeric-ids --protect-args \
  -e "ssh -i $key -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes" \
  "$probe_src/payload/" "$remote:$probe_dst/payload/"

ssh "${ssh_options[@]}" "$remote" \
  "python3 - '$probe_dst/payload' '$expected_sha' '$expected_mtime'" <<'PY'
import hashlib, os, stat, sys
p, expected, expected_mtime = sys.argv[1], sys.argv[2], int(sys.argv[3])
def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
r = os.path.join(p, "regular.bin")
h = os.path.join(p, "hardlink.bin")
l = os.path.join(p, "symlink.bin")
s = os.path.join(p, "sparse.bin")
assert sha(r) == expected, "regular_sha256"
assert os.stat(r).st_ino == os.stat(h).st_ino, "hardlink_inode"
assert os.stat(r).st_nlink == 2, "hardlink_count"
assert stat.S_IMODE(os.stat(r).st_mode) == 0o640, "mode"
assert os.stat(r).st_uid == 12345, "numeric_uid"
assert os.stat(r).st_gid == 23456, "numeric_gid"
assert int(os.stat(r).st_mtime) == expected_mtime, "mtime"
assert os.path.islink(l) and os.readlink(l) == "regular.bin", "symlink"
value = os.stat(s)
assert value.st_size == 10485760, "sparse_logical_size"
print("NAS_PROBE_PASS")
print("regular_sha256=" + expected)
print("hardlink_nlink=" + str(os.stat(r).st_nlink))
print("sparse_allocated_bytes=" + str(value.st_blocks * 512))
print("sparse_layout_preserved=" + str(
    value.st_blocks * 512 < value.st_size).lower())
PY

cleanup
trap - EXIT
echo "PROBE_CLEANUP_PASS"
