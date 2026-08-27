# SPDX-License-Identifier: MPL-2.0

BEGIN {
    FS = "\t"
    if (expected_runs !~ /^[1-9][0-9]*$/ ||
        expected_tuples !~ /^[1-9][0-9]*$/ ||
        expected_processes !~ /^[1-9][0-9]*$/) {
        fail("InvalidExpectedCounts")
    }
}

function fail(message) {
    failed = 1
    print "array-attribution protocol error: " message > "/dev/stderr"
    exit 1
}

{
    if (NF != 22) fail("InvalidResultFieldCount")
    operation = $1
    arm = $2
    median = $3
    digest = $4
    fingerprint = $5
    pairs = $6
    inputs = $7
    bitset = $8
    other = $9
    unmatched_left = $10
    unmatched_right = $11
    conversions = $12
    allocations = $13
    normalizations = $14
    branch = $15
    distinct = $16
    storage = $17

    if ((operation != "pair-or" && operation != "pair-andnot") ||
        arm == "" || median !~ /^[0-9]+$/ ||
        digest !~ /^0x[0-9a-f]+$/ || fingerprint !~ /^0x[0-9a-f]+$/) {
        fail("InvalidResultValue")
    }
    for (field = 6; field <= 14; field++) {
        if ($field !~ /^[0-9]+$/) fail("InvalidNumericMetadata")
    }
    if (distinct !~ /^[01]$/ || storage !~ /^[01]$/) fail("InvalidBooleanMetadata")
    for (field = 18; field <= 22; field++) {
        if ($field !~ /^[0-9]+$/) fail("InvalidSizeDistribution")
    }

    tuple = operation SUBSEP arm
    process_count++
    if (!(tuple in tuple_seen)) {
        tuple_seen[tuple] = 1
        tuple_count++
        tuple_digest[tuple] = digest
        tuple_static[tuple] = fingerprint SUBSEP pairs SUBSEP inputs SUBSEP bitset SUBSEP other \
            SUBSEP unmatched_left SUBSEP unmatched_right SUBSEP conversions SUBSEP allocations \
            SUBSEP normalizations SUBSEP branch SUBSEP distinct SUBSEP storage SUBSEP $18 SUBSEP $19 \
            SUBSEP $20 SUBSEP $21 SUBSEP $22
    } else {
        if (tuple_digest[tuple] != digest) fail("DigestRepeatMismatch")
        current_static = fingerprint SUBSEP pairs SUBSEP inputs SUBSEP bitset SUBSEP other \
            SUBSEP unmatched_left SUBSEP unmatched_right SUBSEP conversions SUBSEP allocations \
            SUBSEP normalizations SUBSEP branch SUBSEP distinct SUBSEP storage SUBSEP $18 SUBSEP $19 \
            SUBSEP $20 SUBSEP $21 SUBSEP $22
        if (tuple_static[tuple] != current_static) fail("MetadataRepeatMismatch")
    }
    tuple_processes[tuple]++

    common = fingerprint SUBSEP pairs SUBSEP inputs SUBSEP bitset SUBSEP other SUBSEP unmatched_left \
        SUBSEP unmatched_right SUBSEP $18 SUBSEP $19 SUBSEP $20 SUBSEP $21 SUBSEP $22
    if (!(operation in operation_seen)) {
        operation_seen[operation] = 1
        operation_common[operation] = common
    } else if (operation_common[operation] != common) {
        fail("ArmInputCountMismatch")
    }

    digest_group = arm ~ /^e[12]-/ ? "endtoend" : "matched"
    digest_key = operation SUBSEP digest_group
    if (!(digest_key in group_seen)) {
        group_seen[digest_key] = 1
        group_digest[digest_key] = digest
    } else if (group_digest[digest_key] != digest) {
        fail("DigestCrossArmMismatch")
    }

    if (arm ~ /^a[123]-/) {
        if (allocations != 0) fail("LayerAAllocated")
        if (distinct != 1) fail("LayerAOutputAliasesInput")
        if (storage != 1) fail("LayerAOutputStorageChanged")
    }
    if (arm == "b3-rawr-no-normalize" && normalizations != 0) fail("B3Normalized")
}

END {
    if (failed) exit 1
    if (process_count != expected_processes) fail("ProcessCountMismatch")
    if (tuple_count != expected_tuples) fail("TupleCountMismatch")
    for (tuple in tuple_seen) {
        if (tuple_processes[tuple] != expected_runs) fail("TupleProcessCountMismatch")
    }
    print "VALIDATED\tprocesses=" process_count "\ttuples=" tuple_count
}
