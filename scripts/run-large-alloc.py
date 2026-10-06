#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Spec 52-01 controller. Only Python's standard library is required."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

PHASES = ["init", "warmup", "timed", "cleanup", "done"]
TUNABLES = {"MALLOC_MMAP_THRESHOLD_": "16384", "MALLOC_TRIM_THRESHOLD_": "0", "MALLOC_TOP_PAD_": "0"}
SIZES = [4096, 16384, 32768, 32769, 65536, 262144, 1048576, 4194304, 16777216]


def parse_trace(text):
    counts = {p: {"mmap": 0, "munmap": 0} for p in ["startup"] + PHASES}
    phase, index = "startup", 0
    for line in text.splitlines():
        if re.search(r"\bwrite\(198,", line):
            match = re.search(r'"RAWR52:(\w+)\\n", (\d+)\)\s+=\s+(\d+)$', line)
            if not match or index >= len(PHASES) or match[1] != PHASES[index] or match[2] != match[3]:
                raise ValueError("MalformedMarker")
            phase = match[1]
            index += 1
        else:
            match = re.search(r"\b(mmap|munmap)\(", line)
            if match:
                if "<unfinished" in line or "= -1" in line or not re.search(r"\)\s+=\s+", line):
                    raise ValueError("IncompleteOrFailedMapping")
                counts[phase][match[1]] += 1
    if index != len(PHASES):
        raise ValueError("MissingMarker")
    return counts


def marker(p):
    n = len("RAWR52:" + p + "\n")
    return f'write(198, "RAWR52:{p}\\n", {n}) = {n}'


def parser_controls():
    lines = []
    for p, n in zip(PHASES, [1, 1, 2, 3, 0]):
        lines.append(marker(p))
        for _ in range(n):
            lines += ["mmap(NULL, 4096, 3, 34, -1, 0) = 0x1000", "munmap(0x1000, 4096) = 0"]
    text = "\n".join(lines)
    result = parse_trace(text)
    for p, n in zip(PHASES, [1, 1, 2, 3, 0]):
        assert result[p] == {"mmap": n, "munmap": n}
    for bad in [text.replace(marker("timed"), ""), text + "\n" + marker("done"),
                text.replace(marker("warmup"), marker("timed")), text.replace("= 13", "= 0")]:
        try:
            parse_trace(bad)
        except ValueError:
            continue
        raise AssertionError("MarkerMutationNotDetected")
    # A deliberately always-zero result must fail the positive oracle.
    zero = {p: {"mmap": 0, "munmap": 0} for p in PHASES}
    assert zero["timed"] != result["timed"]


def parse_worker(text, kind, mode, batch):
    lines = [line.split("\t") for line in text.splitlines()]
    results = [x for x in lines if x[0] == "RESULT"]
    metas = [x for x in lines if x[0] == "META"]
    samples = [x for x in lines if x[0] == "SAMPLE"]
    if len(results) != 1 or len(metas) != 1 or len(samples) != 24:
        raise ValueError("WorkerProtocolCount")
    if results[0][1:3] != [kind, mode] or int(results[0][4]) != batch:
        raise ValueError("WorkerTupleMismatch")
    by_phase = {}
    for phase, n in [("warmup", 3), ("timed", 21)]:
        rows = [x for x in samples if x[1] == phase]
        if [int(x[2]) for x in rows] != list(range(n)):
            raise ValueError("WorkerSampleOrder")
        by_phase[phase] = [[int(v) for v in row[3:]] for row in rows]
        if any(len(s) != 3 or min(s) < 0 for s in by_phase[phase]):
            raise ValueError("WorkerSampleValue")
    median = statistics.median(x[0] for x in by_phase["timed"])
    if median != int(results[0][5]):
        raise ValueError("WorkerMedianMismatch")
    m = metas[0]
    meta = dict(zip(["zig", "arch", "os", "optimize"], m[1:5]))
    meta.update(zip(["page_size", "slab", "ceiling", "fault_source", "bytes", "alignment", "offset_min", "offset_max"], map(int, m[5:])))
    if meta["fault_source"] != 1 or meta["os"] != "linux" or meta["optimize"] != "ReleaseFast":
        raise ValueError("UnsupportedMeasurementEnvironment")
    if meta["bytes"] != int(results[0][3]):
        raise ValueError("WorkerSizeMismatch")
    return {"ns": median / batch, "batch_ns": median, "meta": meta, "samples": by_phase,
            "minor": statistics.median(x[1] for x in by_phase["timed"]) / batch,
            "major": statistics.median(x[2] for x in by_phase["timed"]) / batch}


def execute(command, path, env):
    with path.open("w") as log:
        p = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    text = path.read_text()
    if p.returncode:
        raise RuntimeError(f"exit={p.returncode}: {path}\n{text[-2000:]}")
    return text


def summarize(values):
    return [statistics.median(values), min(values), max(values)]


def write_summary(out, aggregate):
    lines = ["Spec 52-01: untraced timing, separately traced syscall counts", "ms = per-cycle median of process medians [min,max]",
             "Replay offsets are post-timing production observations; bare offsets cover warmup and timed cycles.",
             "full-payload-pages describes the address span, not actual touches in the no-write control.",
             "No cross-host attribution; page touches are not predicted fault counts.", ""]
    for r in aggregate:
        m, lo, hi = [v / 1e6 for v in r["ns"]]
        maps = r["trace"]["timed"]["mmap"] / (21 * r["batch"])
        unmaps = r["trace"]["timed"]["munmap"] / (21 * r["batch"])
        lines.append(f"{r['name']:<48} {m:.6f} [{lo:.6f},{hi:.6f}] minor={r['minor'][0]:.3f} major={r['major'][0]:.3f} mmap={maps:.3f} munmap={unmaps:.3f} full-payload-pages={r['touches']}")
        if "retry" in r:
            m, lo, hi = [v / 1e6 for v in r["retry"]["ns"]]
            lines.append(f"  retry ms={m:.6f} [{lo:.6f},{hi:.6f}]")
        if "reproduction" in r:
            lines.append("  reproduction=" + json.dumps(r["reproduction"]))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"saved {out}/summary.txt")


def worker_protocol_controls():
    text = "META\t0.16.0\tx86_64\tlinux\tReleaseFast\t4096\t65536\t32768\t1\t4096\t1\t0\t0\n"
    for phase, count in [("warmup", 3), ("timed", 21)]:
        for i in range(count):
            text += f"SAMPLE\t{phase}\t{i}\t2000000\t0\t0\n"
    text += "RESULT\tsmp\twrite\t4096\t1\t2000000\n"
    assert parse_worker(text, "smp", "write", 1)["ns"] == 2000000
    mutations = [text.replace("RESULT\tsmp", "RESULT\tlibc"),
                 text.replace("SAMPLE\ttimed\t20", "SAMPLE\ttimed\t19"),
                 text.replace("RESULT\tsmp\twrite\t4096\t1\t2000000", "RESULT\tsmp\twrite\t4096\t1\t1"),
                 text + "RESULT\tsmp\twrite\t4096\t1\t2000000\n"]
    for bad in mutations:
        try:
            parse_worker(bad, "smp", "write", 1)
        except ValueError:
            continue
        raise AssertionError("WorkerMutationNotDetected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report-only", type=Path, help="regenerate summary from retained results.json without measuring")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--induce", action="store_true", help="primary Linux/glibc host only")
    args = parser.parse_args()
    parser_controls()
    worker_protocol_controls()
    if args.self_test:
        print("trace parser: positive attribution, always-zero mutation, malformed markers OK; worker tuple/count/order/median controls OK")
        return
    if args.report_only:
        write_summary(args.report_only, json.loads((args.report_only / "results.json").read_text()))
        return
    if args.runs < 5 or args.runs % 2 == 0:
        parser.error("runs must be odd and >=5")
    out = args.output or Path("misc") / time.strftime("large-alloc-%Y%m%d-%H%M%S", time.gmtime())
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    inherited = {k: v for k, v in env.items() if k.startswith("MALLOC_") or k in ("GLIBC_TUNABLES", "LD_PRELOAD", "LD_AUDIT")}
    for k in inherited:
        env.pop(k)
    if not args.skip_build:
        execute(["zig", "build", "bench-large-alloc", "-Dcpu=native"], out / "build.log", env)
    workers = {"bare": Path("zig-out/bin/bench_large_alloc").resolve(), "replay": Path("zig-out/bin/bench_large_alloc_replay").resolve()}
    provenance = {"removed_environment": inherited, "induction_tunables": TUNABLES if args.induce else None,
                  "protocol": "3 warmups, 21 timed, median; >=5 fresh processes; timings untraced",
                  "fault_source": "RAWR_RESIDENCY_FAULT_LINUX_RUSAGE", "uname": list(os.uname()),
                  "glibc": os.confstr("CS_GNU_LIBC_VERSION"),
                  "worker_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in workers.items()},
                  "huge_pages": {p: Path(p).read_text() if Path(p).exists() else "unavailable" for p in
                                 ["/sys/kernel/mm/transparent_hugepage/enabled", "/sys/kernel/mm/transparent_hugepage/defrag"]}}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    execute(["lscpu"], out / "cpu.txt", env)
    # Real syscall control: initialization/one warmup/two timed/three cleanup mappings.
    trace_path = out / "control.trace"
    execute(["strace", "-f", "-s", "128", "-e", "trace=mmap,munmap,write", "-o", str(trace_path),
             str(workers["bare"]), "--trace-control"], out / "control.log", env)
    control = parse_trace(trace_path.read_text())
    for phase, n in zip(PHASES, [1, 1, 2, 3, 0]):
        if control[phase] != {"mmap": n, "munmap": n}:
            raise ValueError(f"TraceAttributionControl: {control}")
    (out / "controls.json").write_text(json.dumps(control, indent=2))
    cells = [("replay", kind, mode, 1, False) for mode in ["serialize", "to_array_alloc"] for kind in ["smp", "libc"]]
    cells += [("bare", kind, mode, size, False) for size in SIZES for mode in ["write", "no_write", "retained"] for kind in ["smp", "libc"]]
    cells += [("bare", "page", mode, 4194304, False) for mode in ["write", "no_write", "retained"]]
    if args.induce:
        cells += [("bare", "libc", mode, size, True) for size in SIZES for mode in ["write", "no_write"]]
    aggregate = []
    for layer, kind, mode, size, induced in cells:
        name = f"{layer}-{kind}-{mode}-{size}" + ("-induced" if induced else "")
        cell_env = dict(env, **(TUNABLES if induced else {}))
        batch = 1
        if layer == "bare":
            # Every candidate count is calibrated in a disposable worker.
            for attempt in range(8):
                command = [str(workers[layer]), kind, mode, str(size), str(batch)]
                trial = parse_worker(execute(command, out / f"{name}-cal{attempt}.log", cell_env), kind, mode, batch)
                if trial["batch_ns"] >= 2_000_000:
                    break
                batch = max(batch + 1, math.ceil(batch * 2_500_000 / max(1, trial["batch_ns"])))
                if batch > 1_048_576:
                    raise ValueError("CalibrationLimit")
            else:
                raise ValueError("CalibrationFailed")
        command = [str(workers[layer]), kind, mode, str(size), str(batch)]
        def run_set(suffix):
            return [parse_worker(execute(command, out / f"{name}-{suffix}{i}.log", cell_env), kind, mode, batch) for i in range(args.runs)]
        processes = run_set("run")
        # Trace the same batch/protocol in another process, never timing under strace.
        trace_path = out / f"{name}.trace"
        execute(["strace", "-f", "-s", "128", "-e", "trace=mmap,munmap,write", "-o", str(trace_path)] + command + ["--trace"], out / f"{name}-trace.log", cell_env)
        trace = parse_trace(trace_path.read_text())
        record = {"name": name, "layer": layer, "kind": kind, "mode": mode, "size": size, "induced": induced,
                  "batch": batch, "ns": summarize([p["ns"] for p in processes]),
                  "minor": summarize([p["minor"] for p in processes]), "major": summarize([p["major"] for p in processes]),
                  "processes": processes, "trace": trace}
        if layer == "bare" and any(min(s[0] for s in p["samples"]["timed"]) < 1_000_000 for p in processes):
            raise ValueError(f"BatchBelowOneMillisecond: {name}")
        if any(p["meta"]["ceiling"] != 32768 for p in processes):
            raise ValueError("SweepRequiresNewBoundary")
        record["touches"] = [min(math.ceil((p["meta"]["offset_min"] + p["meta"]["bytes"]) / p["meta"]["page_size"]) for p in processes),
                             max(math.ceil((p["meta"]["offset_max"] + p["meta"]["bytes"]) / p["meta"]["page_size"]) for p in processes)]
        aggregate.append(record)
        (out / "results.json").write_text(json.dumps(aggregate, indent=2))
        print(f"{name}: {record['ns'][0]/1e6:.6f} ms faults={record['minor'][0]:.3f} batch={batch}", flush=True)
        # Finish both allocator cells before applying the per-row reproduction rule.
        if layer == "replay" and kind == "libc":
            a, b = aggregate[-2:]
            def verdict(a, b):
                if a["ns"][1] / b["ns"][2] > 1.10: return "present"
                if a["ns"][2] / b["ns"][1] <= 1.10: return "absent"
                return "unresolved"
            first = verdict(a, b)
            if first == "unresolved":
                # One pre-registered rerun of both cells, retain both sets separately.
                for r in [a, b]:
                    cmd = [str(workers["replay"]), r["kind"], mode, "1", "1"]
                    ps = [parse_worker(execute(cmd, out / f"{r['name']}-retry{i}.log", env), r["kind"], mode, 1) for i in range(args.runs)]
                    r["retry"] = {"ns": summarize([p["ns"] for p in ps]), "processes": ps}
                final = verdict(a["retry"], b["retry"])
            else:
                final = first
            b["reproduction"] = {"first": first, "final": final, "scope": "this host only; WSL2 transfer unverified"}
            (out / "results.json").write_text(json.dumps(aggregate, indent=2))
    if len(aggregate) != len(cells):
        raise ValueError("CellCountMismatch")
    write_summary(out, aggregate)


if __name__ == "__main__":
    main()
