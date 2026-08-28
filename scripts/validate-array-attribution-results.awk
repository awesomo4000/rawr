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
    if (NF != 40) fail("InvalidResultFieldCount")
    dataset = $1
    operation = $2
    arm = $3
    median = $4
    digest = $5
    fingerprint = $6

    if ((dataset != "uscensus2000" && dataset != "census1881" && dataset != "wikileaks-noquotes") ||
        (operation != "pair-or" && operation != "pair-andnot") ||
        (arm != "a1-rawr-scalar" && arm != "a2-croaring-scalar" &&
         arm != "c1-bulk-tail" && arm != "c2-branchy" && arm != "c3-branchy-bulk-tail") ||
        median !~ /^[0-9]+$/ || digest !~ /^0x[0-9a-f]+$/ ||
        fingerprint !~ /^0x[0-9a-f]+$/) {
        fail("InvalidResultValue")
    }
    for (field = 7; field <= 15; field++) {
        if ($field !~ /^[0-9]+$/) fail("InvalidNumericMetadata")
    }
    if ($16 == "") fail("MissingBranchMetadata")
    if ($17 !~ /^[01]$/ || $18 !~ /^[01]$/) fail("InvalidBooleanMetadata")
    for (field = 19; field <= 40; field++) {
        if ($field !~ /^[0-9]+$/) fail("InvalidDiagnosticMetadata")
    }

    tuple = dataset SUBSEP operation SUBSEP arm
    process_count++
    if (!(tuple in tuple_seen)) {
        tuple_seen[tuple] = 1
        tuple_count++
        tuple_digest[tuple] = digest
        tuple_static[tuple] = staticFields()
    } else {
        if (tuple_digest[tuple] != digest) fail("DigestRepeatMismatch")
        if (tuple_static[tuple] != staticFields()) fail("MetadataRepeatMismatch")
    }
    tuple_processes[tuple]++

    common_key = dataset SUBSEP operation
    common = fingerprint
    for (field = 7; field <= 12; field++) common = common SUBSEP $field
    for (field = 19; field <= 40; field++) common = common SUBSEP $field
    if (!(common_key in common_seen)) {
        common_seen[common_key] = 1
        common_static[common_key] = common
    } else if (common_static[common_key] != common) {
        fail("ArmInputOrDiagnosticMismatch")
    }

    digest_key = dataset SUBSEP operation
    if (!(digest_key in digest_seen)) {
        digest_seen[digest_key] = 1
        group_digest[digest_key] = digest
    } else if (group_digest[digest_key] != digest) {
        fail("DigestCrossArmMismatch")
    }

    if ($14 != 0) fail("LayerAAllocated")
    if ($17 != 1) fail("LayerAOutputAliasesInput")
    if ($18 != 1) fail("LayerAOutputStorageChanged")
}

function staticFields(    value, field) {
    value = $5
    for (field = 6; field <= 40; field++) value = value SUBSEP $field
    return value
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
