<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 47-02: Runtime cells, the evidence table, and the README

Toplevel: [47-portability-matrix.md](47-portability-matrix.md).
Gated on: [47-01](47-01-compile-matrix.md) complete.

Produces the deliverable. **Provisioning is owner-handled and is not a blocker** — a cell with no host
stays `compiles`, and the chunk completes with partial runtime coverage.

## 1. Runtime set, per available host

1. `zig build test`
2. `zig build test64`
3. **`zig build check-package`** — the allowlist-only consumer builds *and runs*. Closest proxy for "a
   real user on this platform can use rawr", and **the check that settles toplevel §1's open question**:
   whether a consumer resolving the shipped `build.zig` ever reaches its OpenBSD or FreeBSD branches.
   Run it there first if either host is available.

`difftest` / `difftest64` are **Tier 2** — they link CRoaring. Run where they work; **absence is a testing
gap, never a library defect**, and it may not be reported as an unsupported platform.

## 2. The evidence table

In `docs/`, repo-only. Every cell carries exactly one toplevel §6 status: `verified`, `compiles`,
`tooling-gap` (with the reason), `broken` (with the error), or `not targetable`.

**Keyed by target triple, not OS family.** Running `windows-gnu` does not make `windows-msvc` `verified`;
`linux-gnu` says nothing about `linux-musl`. An executed cell is `verified` and its unexecuted ABI
siblings stay `compiles`. The 12 host families are provisioning context; the **16 target cells** are the
evidence unit.

**The Linux/x86_64 cell is recorded as WSL2** until native Linux runs. It is a Linux kernel under a
Windows host, and `52-00` Part B showed it producing a 2.47x different result for identical code — a
timing finding, not a correctness one, so the cell is genuine evidence for buildability and test-passing.
But "Linux/x86_64 verified" unqualified would rest on a virtualized environment.

**`52-00` Part A upgrades this cell when it runs.** That is a one-line update here, and it is the only
coupling between the two campaigns. **`47-02` does not wait for it.**

## 3. The README

Update the support statement to match the table. **This is the point of the exercise** — the arch/OS
matrix currently has no statement, and it must not acquire an optimistic one.

Per spec 41's language discipline: state **what is tested**, no performance claims, no implied guarantees
for untested cells.

- **`verified` and `compiles` are different words and stay different.** A cell that only cross-compiles is
  never described as supported.
- **Tier 2 gaps do not appear as unsupported platforms.** If `difftest` cannot run somewhere, that is our
  testing limitation, and saying otherwise misreports the library's state to users.
- **Name the Linux/x86_64 cell as WSL2** while that is what it is.

## 4. What this chunk cannot conclude

An unprovisioned cell is **unknown**, not working and not broken. `compiles` carries exactly the
information it names. **The table's value is that it can say "we do not know"** — a matrix that reads as
uniformly reassuring would be less useful than none.

## Acceptance

- §1 runtime set run on **every host the owner provides**; unavailable cells left at `compiles` and the
  chunk **completes anyway**.
- `check-package` run on OpenBSD and/or FreeBSD **if either is available**, with toplevel §1's consumer-path
  question answered or explicitly left open.
- Evidence table in `docs/`, **keyed by target triple**, every one of the 16 cells plus the two
  baseline-feature cells carrying exactly one §6 status, with reasons and errors where the status requires
  them.
- **No ABI sibling promoted by association.**
- Linux/x86_64 recorded as **WSL2**, with a note that `52-00` Part A upgrades it.
- README support statement matches the table, **`verified` and `compiles` distinguished**, no performance
  claims, **Tier 2 gaps not presented as unsupported platforms**.
- `check-docs` green — the README change is exactly the kind of drift it exists to catch.
- Existing suites plus `check-32`, `check-portability`, `check-package` green on the dev host.

## Estimate

**S/M** — the running is quick where hosts exist. The care is in the table and the README wording, which
is where an honest result gets quietly upgraded into a reassuring one.
