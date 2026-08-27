# SPDX-License-Identifier: MPL-2.0

BEGIN {
    FS = "\t"
    split("0 1-2 3-6 7-12 13-32 33-128 129+", bands, " ")
    split("1 2 4 6 8 12", q3_cards, " ")
}

$1 == "MIXED_AGG" {
    key = $2 SUBSEP $3 SUBSEP $4 SUBSEP $5
    mixed_batch[key] = $6
    mixed_median[key] = $7
    mixed_min[key] = $8
    mixed_max[key] = $9
    next
}

$1 == "MIXED_META" {
    mixed_count = $2
    mixed_median_cardinality = $3
    mixed_p99 = $4
    mixed_hash = $5
    next
}

$1 == "MIXED_ACCOUNT" {
    key = $2 SUBSEP $3
    account_count[key] = $4
    account_alloc[key] = $5
    account_free[key] = $6
    account_resize[key] = $7
    account_requested[key] = $8
    account_create[key] = $9
    account_build[key] = $10
    account_serialize[key] = $11
    account_deserialize[key] = $12
    account_peak_sum[key] = $13
    account_peak_max[key] = $14
    account_serialized[key] = $15
    next
}

$1 == "AGG" {
    key = $2 SUBSEP $3 SUBSEP $4 SUBSEP $5
    sweep_batch[key] = $6
    sweep_median[key] = $7
    next
}

$1 == "ACCOUNT" {
    key = $2 SUBSEP $3 SUBSEP $4 SUBSEP $5
    sweep_create[key] = $11
    sweep_build[key] = $12
    sweep_serialized[key] = $17
    next
}

function mixed_key(cell, band, implementation, allocator) {
    return cell SUBSEP band SUBSEP implementation SUBSEP allocator
}

function account_key(name, allocator) {
    return name SUBSEP allocator
}

function sweep_key(cardinality, implementation, allocator) {
    return "spread" SUBSEP cardinality SUBSEP implementation SUBSEP allocator
}

function mixed_mean_ns(key) {
    return mixed_median[key] / mixed_batch[key]
}

function print_total_cells(    i, implementation, allocator, key, label) {
    print "Measured mixed-corpus totals"
    print "tuple              count    total ms [min,max]          mean ns/bitmap"
    for (i = 1; i <= 5; i++) {
        if (i == 1) { implementation = "rawr"; allocator = "smp" }
        if (i == 2) { implementation = "rawr"; allocator = "libc" }
        if (i == 3) { implementation = "croaring"; allocator = "libc" }
        if (i == 4) { implementation = "reference"; allocator = "smp" }
        if (i == 5) { implementation = "reference"; allocator = "libc" }
        key = mixed_key("total", "total", implementation, allocator)
        label = implementation "/" allocator
        printf "%-18s %6d  %10.3f [%8.3f,%8.3f]  %14.2f\n", \
            label, mixed_batch[key], mixed_median[key] / 1000000, mixed_min[key] / 1000000, \
            mixed_max[key] / 1000000, mixed_mean_ns(key)
    }
    print ""
}

function print_projection(allocator,    i, band, key, total_key, mean_ns, projected, projected_total, measured, residual, tiny_time) {
    total_key = mixed_key("total", "total", "rawr", allocator)
    measured = mixed_median[total_key]
    projected_total = 0
    tiny_time = 0
    for (i = 2; i <= 7; i++) {
        band = bands[i]
        key = mixed_key("band", band, "rawr", allocator)
        mean_ns = mixed_mean_ns(key)
        projected = mean_ns * account_count[account_key(band, allocator)]
        projected_time[band] = projected
        projected_total += projected
        if (i <= 4) tiny_time += projected
    }

    print "Projected rawr/" allocator " band attribution"
    print "band       count  corpus share  mean ns/bitmap  projected ms  projected time share"
    printf "%-9s %6d  %11.3f%%  %14s  %12.3f  %19.3f%%\n", \
        "0", 0, 0, "N/A", 0, 0
    for (i = 2; i <= 7; i++) {
        band = bands[i]
        key = mixed_key("band", band, "rawr", allocator)
        printf "%-9s %6d  %11.3f%%  %14.2f  %12.3f  %19.3f%%\n", \
            band, account_count[account_key(band, allocator)], \
            100 * account_count[account_key(band, allocator)] / mixed_count, mixed_mean_ns(key), \
            projected_time[band] / 1000000, 100 * projected_time[band] / projected_total
    }
    residual = projected_total - measured
    printf "projected total: %.3f ms\n", projected_total / 1000000
    printf "measured total:  %.3f ms\n", measured / 1000000
    printf "signed projection residual: %+.3f ms (%+.3f%% of measured)\n", \
        residual / 1000000, 100 * residual / measured
    printf "tiny-tail projected time share (1-12): %.3f%%\n", 100 * tiny_time / projected_total
    print ""
}

function print_accounting(allocator,    i, band, key, total_key, tiny_requested, tiny_alloc, tiny_serialized) {
    total_key = account_key("total", allocator)
    tiny_requested = 0
    tiny_alloc = 0
    tiny_serialized = 0

    print "Measured rawr/" allocator " byte and allocation attribution"
    print "band       count  alloc calls  alloc share  requested bytes  byte share  serialized bytes  peak max"
    printf "%-9s %6d  %11d  %10.3f%%  %15d  %9.3f%%  %16d  %8d\n", \
        "0", 0, 0, 0, 0, 0, 0, 0
    for (i = 2; i <= 7; i++) {
        band = bands[i]
        key = account_key(band, allocator)
        printf "%-9s %6d  %11d  %10.3f%%  %15d  %9.3f%%  %16d  %8d\n", \
            band, account_count[key], account_alloc[key], 100 * account_alloc[key] / account_alloc[total_key], \
            account_requested[key], 100 * account_requested[key] / account_requested[total_key], \
            account_serialized[key], account_peak_max[key]
        if (i <= 4) {
            tiny_requested += account_requested[key]
            tiny_alloc += account_alloc[key]
            tiny_serialized += account_serialized[key]
        }
    }
    printf "total     %6d  %11d              %15d             %16d  %8d\n", \
        account_count[total_key], account_alloc[total_key], account_requested[total_key], \
        account_serialized[total_key], account_peak_max[total_key]
    printf "tiny-tail measured requested-byte share (1-12): %.3f%%\n", \
        100 * tiny_requested / account_requested[total_key]
    printf "tiny-tail measured allocation-call share (1-12): %.3f%%\n", \
        100 * tiny_alloc / account_alloc[total_key]
    printf "tiny-tail measured serialized-byte share (supporting): %.3f%%\n", \
        100 * tiny_serialized / account_serialized[total_key]
    print ""
}

function print_q3(    i, card, rawr_time_key, ref_time_key, rawr_account_key, ref_account_key, time_ratio, byte_ratio, previous_time, previous_byte, time_monotonic, byte_monotonic) {
    print "Q3 spread-shape gap to the plain-list references (rawr/SMP)"
    print "card  time ratio  byte ratio  create bytes  create->build bytes"
    time_monotonic = 1
    byte_monotonic = 1
    for (i = 1; i <= 6; i++) {
        card = q3_cards[i]
        rawr_time_key = sweep_key(card, "rawr", "smp")
        ref_time_key = sweep_key(card, "reference", "smp")
        rawr_account_key = sweep_key(card, "rawr", "smp")
        ref_account_key = sweep_key(card, "reference", "smp")
        time_ratio = (sweep_median[rawr_time_key] / sweep_batch[rawr_time_key]) / \
            (sweep_median[ref_time_key] / sweep_batch[ref_time_key])
        byte_ratio = sweep_serialized[rawr_account_key] / sweep_serialized[ref_account_key]
        printf "%4d  %9.2fx  %9.2fx  %12.0f  %19.0f\n", card, time_ratio, byte_ratio, \
            sweep_create[rawr_account_key], sweep_build[rawr_account_key] - sweep_create[rawr_account_key]
        if (i > 1 && time_ratio < previous_time) time_monotonic = 0
        if (i > 1 && byte_ratio < previous_byte) byte_monotonic = 0
        previous_time = time_ratio
        previous_byte = byte_ratio
    }
    print "time ratio monotonic nondecreasing: " (time_monotonic ? "yes" : "no")
    print "byte ratio monotonic nondecreasing: " (byte_monotonic ? "yes" : "no")
    print ""
}

END {
    print "Q5 mixed-corpus attribution and decision inputs"
    print "================================================"
    print "Corpus: count=" mixed_count ", shape=spread, median=" mixed_median_cardinality ", p99=" mixed_p99
    print "Cardinality hash: " mixed_hash
    print "Time shares below are projected from independent fresh-process band cells."
    print "Requested-byte and allocation-call shares are measured in the untimed accounting pass."
    print "No automatic verdict or threshold is applied; the owner decides from Q3 and Q5."
    print ""

    print_total_cells()
    print_projection("smp")
    print_projection("libc")
    print_accounting("smp")
    print_accounting("libc")
    print_q3()

    print "Scope: RoaringBitmap/u32 only. Roaring64Bitmap is deferred to the 10-21-bench64 work."
}
