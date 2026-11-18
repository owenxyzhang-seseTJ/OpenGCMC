# Host-grid disk cache

The POSIX writer streams native-endian PMCHG061 fields into an exclusively created sibling file with a short name. It checks field lengths, writes and hashes blocks, flushes, syncs and closes the file, then publishes it with rename. Failures before publication preserve a previous cache, including failures after a partial payload has been written. Readers open the old or new inode and check its physics fingerprint, dimensions and payload checksum. The fingerprint determines which physical model is reusable when different jobs target the same filename. Use separate names per system to avoid redundant rebuilding. Linux temporary files are private to the creating account (0600). A killed writer may leave an unused temporary file. POSIX rename provides atomic visibility under the target filesystem's rename semantics. The file is synced before publication; retention of the directory entry across a power outage depends on the filesystem.

The reported HPC segmentation fault was localized to the old save routine. No faulting instruction, core file or matching runtime was supplied. Local testing reproduced a different, specific save defect: the former path-plus-PID-plus-clock temporary filename exceeds NAME_MAX even when the requested filename itself is valid. Linux host-grid writing now uses libc/POSIX, including directory creation and atomic replacement. The old binary referenced eight GLIBCXX filesystem symbols, the isolation build referenced none. This observation does not establish an ABI mismatch on the HPC node.

## Reproduction

From the release root, after building the native library:

```sh
python -m pytest -q tests07/test_host_cache.py
python tools07/audit_host_cache.py --input real.json --mesh 64 --workdir . --output host_cache_hpc.json
```

Place `--workdir` on the filesystem used in production. The tool uses three independent processes with the same seed. It compares cache OFF, first construction and persistence, and warm reuse without grid reconstruction. It checks hashes of coordinates, count and energy histories and move counters; the final target energy must also agree. With no input, it uses a charged synthetic 4,416-site fixture, 12-site molecule, first five guest molecules, 10 trials, and rectangular kmax 13. It creates and removes a private run directory under workdir.

The writer isolation measurements, both directions of 0.6 file compatibility, 255-byte filename and transient failure tests are in the adjacent JSON files. Compatibility is on the same native endian/float representation. The sanitizer check also covered an empty host, charged and uncharged-site fields in the existing grid audit, low-temperature nodes, corrupt checksums and temperature-dependent queries. LeakSanitizer was disabled because it cannot inspect the traced environment.

For the synchronized concurrency test (give the program a path that does not already exist):

```sh
g++ -std=c++17 -O1 -g -pthread -fno-omit-frame-pointer -fsanitize=address,undefined \
  -fno-sanitize-recover=all -Inative native/host_grid.cpp native/kernels.cpp \
  tests07/host_cache_concurrent.cpp -o /tmp/poremc-host-cache-race
scratch_dir=$(mktemp -d)
ASAN_OPTIONS=detect_leaks=0 /tmp/poremc-host-cache-race "$scratch_dir/grid.bin"
```
