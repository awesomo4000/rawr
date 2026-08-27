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
    print "real-data protocol error: " message > "/dev/stderr"
    exit 1
}

{
    if (NF != 12) fail("InvalidResultFieldCount")
    dataset = $1
    operation = $2
    implementation = $3
    denominator = $4
    median = $5
    digest = $6
    fingerprint = $7
    source_cardinality = $8
    arrays = $9
    bitsets = $10
    runs = $11
    serialized_bytes = $12

    if (dataset == "" || operation == "" ||
        (implementation != "rawr" && implementation != "croaring") ||
        denominator !~ /^[1-9][0-9]*$/ || median !~ /^[0-9]+$/ ||
        digest !~ /^0x[0-9a-f]+$/ || fingerprint !~ /^0x[0-9a-f]+$/ ||
        source_cardinality !~ /^[0-9]+$/ || arrays !~ /^[0-9]+$/ ||
        bitsets !~ /^[0-9]+$/ || runs !~ /^[0-9]+$/ ||
        serialized_bytes !~ /^[0-9]+$/) {
        fail("InvalidResultValue")
    }

    tuple = dataset SUBSEP operation SUBSEP implementation
    row = dataset SUBSEP operation
    setup = dataset SUBSEP implementation
    process_count++
    if (!(tuple in tuple_seen)) {
        tuple_seen[tuple] = 1
        tuple_count++
        tuple_denominator[tuple] = denominator
        tuple_digest[tuple] = digest
        tuple_serialized[tuple] = serialized_bytes
    } else {
        if (tuple_denominator[tuple] != denominator) fail("DenominatorRepeatMismatch")
        if (tuple_digest[tuple] != digest) fail("DigestRepeatMismatch")
        if (tuple_serialized[tuple] != serialized_bytes) fail("SerializedBytesRepeatMismatch")
    }
    tuple_processes[tuple]++
    row_seen[row] = 1
    row_digest[row SUBSEP implementation] = digest
    row_impl_seen[row SUBSEP implementation] = 1

    if (!(dataset in dataset_seen)) {
        dataset_seen[dataset] = 1
        dataset_fingerprint[dataset] = fingerprint
        dataset_cardinality[dataset] = source_cardinality
    } else {
        if (dataset_fingerprint[dataset] != fingerprint) fail("CorpusFingerprintMismatch")
        if (dataset_cardinality[dataset] != source_cardinality) fail("SourceCardinalityMismatch")
    }

    histogram = arrays SUBSEP bitsets SUBSEP runs
    if (!(setup in setup_seen)) {
        setup_seen[setup] = 1
        setup_histogram[setup] = histogram
    } else if (setup_histogram[setup] != histogram) {
        fail("HistogramRepeatMismatch")
    }
}

END {
    if (failed) exit 1
    if (process_count != expected_processes) fail("ProcessCountMismatch")
    if (tuple_count != expected_tuples) fail("TupleCountMismatch")
    for (tuple in tuple_seen) {
        if (tuple_processes[tuple] != expected_runs) fail("TupleProcessCountMismatch")
    }
    for (row in row_seen) {
        rawr_key = row SUBSEP "rawr"
        croaring_key = row SUBSEP "croaring"
        if (!(rawr_key in row_impl_seen) || !(croaring_key in row_impl_seen)) {
            fail("MissingImplementation")
        }
        if (row_digest[rawr_key] != row_digest[croaring_key]) {
            fail("DigestCrossImplementationMismatch")
        }
    }
    print "VALIDATED\tprocesses=" process_count "\ttuples=" tuple_count
}
