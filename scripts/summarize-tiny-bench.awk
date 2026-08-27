# SPDX-License-Identifier: MPL-2.0

BEGIN {
    FS = "\t"
    split("localized spread one-per-container", shapes, " ")
    split("0 1 2 4 6 8 12 16 20 32 64 128", cards, " ")
}

$1 == "AGG" {
    key = $2 SUBSEP $3 SUBSEP $4 SUBSEP $5
    batch[key] = $6
    median[key] = $7
    minimum[key] = $8
    maximum[key] = $9
    measured[$2 SUBSEP $3] = 1
    next
}

$1 == "ACCOUNT" {
    key = $2 SUBSEP $3 SUBSEP $4 SUBSEP $5
    accounting_count[key] = $6
    alloc_mean[key] = $7
    free_mean[key] = $8
    resize_mean[key] = $9
    requested_mean[key] = $10
    create_live[key] = $11
    build_live[key] = $12
    serialize_live[key] = $13
    deserialize_live[key] = $14
    peak_mean[key] = $15
    peak_max[key] = $16
    serialized_mean[key] = $17
    histogram[key] = $18
    next
}

function tuple_key(shape, card, implementation, allocator) {
    return shape SUBSEP card SUBSEP implementation SUBSEP allocator
}

function ns_per_lifecycle(key) {
    return median[key] / batch[key]
}

function min_per_lifecycle(key) {
    return minimum[key] / batch[key]
}

function max_per_lifecycle(key) {
    return maximum[key] / batch[key]
}

function ratio_at(shape, card, curve,    rawr_key, ref_key) {
    if (curve == "time-smp") {
        rawr_key = tuple_key(shape, card, "rawr", "smp")
        ref_key = tuple_key(shape, card, "reference", "smp")
        return ns_per_lifecycle(rawr_key) / ns_per_lifecycle(ref_key)
    }
    if (curve == "time-libc") {
        rawr_key = tuple_key(shape, card, "rawr", "libc")
        ref_key = tuple_key(shape, card, "reference", "libc")
        return ns_per_lifecycle(rawr_key) / ns_per_lifecycle(ref_key)
    }
    rawr_key = tuple_key(shape, card, "rawr", "smp")
    ref_key = tuple_key(shape, card, "reference", "smp")
    return serialized_mean[rawr_key] / serialized_mean[ref_key]
}

function curve_complete(shape, curve,    i, card, rawr_key, ref_key) {
    for (i = 1; i <= 12; i++) {
        card = cards[i]
        if (curve == "time-smp") {
            rawr_key = tuple_key(shape, card, "rawr", "smp")
            ref_key = tuple_key(shape, card, "reference", "smp")
            if (!(rawr_key in median) || !(ref_key in median)) return 0
        } else if (curve == "time-libc") {
            rawr_key = tuple_key(shape, card, "rawr", "libc")
            ref_key = tuple_key(shape, card, "reference", "libc")
            if (!(rawr_key in median) || !(ref_key in median)) return 0
        } else {
            rawr_key = tuple_key(shape, card, "rawr", "smp")
            ref_key = tuple_key(shape, card, "reference", "smp")
            if (!(rawr_key in serialized_mean) || !(ref_key in serialized_mean)) return 0
        }
    }
    return 1
}

function sustained_crossover(shape, curve,    start, later, ok, card) {
    if (!curve_complete(shape, curve)) return "incomplete"
    for (start = 2; start <= 12; start++) {
        ok = 1
        for (later = start; later <= 12; later++) {
            if (ratio_at(shape, cards[later], curve) > 2.0) {
                ok = 0
                break
            }
        }
        if (ok) return cards[start]
    }
    return "none in range"
}

function print_timing(shape,    i, card, rs, fs, rl, fl, cr, ratio_smp, ratio_libc, ratio_cr) {
    print "Timing curve: " shape
    print "card  rawr/smp ns [min,max]       ref/smp ns   ratio    rawr/libc ns [min,max]      ref/libc ns  ratio    CR/libc ns  rawr/CR"
    for (i = 1; i <= 12; i++) {
        card = cards[i]
        rs = tuple_key(shape, card, "rawr", "smp")
        fs = tuple_key(shape, card, "reference", "smp")
        rl = tuple_key(shape, card, "rawr", "libc")
        fl = tuple_key(shape, card, "reference", "libc")
        cr = tuple_key(shape, card, "croaring", "libc")
        if (!(rs in median)) continue
        ratio_smp = ns_per_lifecycle(rs) / ns_per_lifecycle(fs)
        ratio_libc = ns_per_lifecycle(rl) / ns_per_lifecycle(fl)
        ratio_cr = ns_per_lifecycle(rl) / ns_per_lifecycle(cr)
        printf "%4d  %9.2f [%7.2f,%7.2f]  %10.2f  %6.2fx  %10.2f [%7.2f,%7.2f]  %11.2f  %6.2fx  %10.2f  %6.2fx\n", \
            card, ns_per_lifecycle(rs), min_per_lifecycle(rs), max_per_lifecycle(rs), \
            ns_per_lifecycle(fs), ratio_smp, ns_per_lifecycle(rl), min_per_lifecycle(rl), \
            max_per_lifecycle(rl), ns_per_lifecycle(fl), ratio_libc, ns_per_lifecycle(cr), ratio_cr
    }
    print "crossover_time_smp=" sustained_crossover(shape, "time-smp")
    print "crossover_time_libc=" sustained_crossover(shape, "time-libc")
    print "crossover_bytes=" sustained_crossover(shape, "bytes")
    print ""
}

function print_bytes(shape,    i, card, rawr_key, cr_key, ref_key, rawr_bytes, cr_bytes, ref_bytes) {
    print "Portable bytes and plain-list gap: " shape
    print "card  rawr bytes  CRoaring bytes  plain-list bytes  rawr/ref  rawr-ref bytes"
    for (i = 1; i <= 12; i++) {
        card = cards[i]
        rawr_key = tuple_key(shape, card, "rawr", "smp")
        cr_key = tuple_key(shape, card, "croaring", "libc")
        ref_key = tuple_key(shape, card, "reference", "smp")
        if (!(rawr_key in serialized_mean)) continue
        rawr_bytes = serialized_mean[rawr_key]
        cr_bytes = serialized_mean[cr_key]
        ref_bytes = serialized_mean[ref_key]
        printf "%4d  %10.2f  %14.2f  %16.2f  %7.2fx  %+14.2f\n", \
            card, rawr_bytes, cr_bytes, ref_bytes, rawr_bytes / ref_bytes, rawr_bytes - ref_bytes
    }
    print ""
}

function print_accounting(shape,    i, j, card, impl, allocator, key, label, delta) {
    print "Lifecycle accounting: " shape
    print "card  tuple              alloc   free resize requested  create->build build_live serialize deserialize peak_mean peak_max histogram"
    for (i = 1; i <= 12; i++) {
        card = cards[i]
        for (j = 1; j <= 5; j++) {
            if (j == 1) { impl = "rawr"; allocator = "smp" }
            if (j == 2) { impl = "rawr"; allocator = "libc" }
            if (j == 3) { impl = "croaring"; allocator = "libc" }
            if (j == 4) { impl = "reference"; allocator = "smp" }
            if (j == 5) { impl = "reference"; allocator = "libc" }
            key = tuple_key(shape, card, impl, allocator)
            if (!(key in accounting_count) || !(shape SUBSEP card in measured)) continue
            label = impl "/" allocator
            delta = build_live[key] - create_live[key]
            printf "%4d  %-17s %6.2f %6.2f %6.2f %9.2f %13.2f %10.2f %9.2f %11.2f %9.2f %8.0f %s\n", \
                card, label, alloc_mean[key], free_mean[key], resize_mean[key], requested_mean[key], \
                delta, build_live[key], serialize_live[key], deserialize_live[key], peak_mean[key], \
                peak_max[key], histogram[key]
        }
    }
    print ""
}

END {
    print "Q1-Q4 tiny-bitmap cardinality sweep"
    print "===================================="
    print "Timing values are medians of fresh-process medians of whole-pool means."
    print "Ratios divide those means; they are not means of per-fixture ratios."
    print "No verdict: Q5 is deferred to spec 48-02."
    print ""

    for (shape_index = 1; shape_index <= 3; shape_index++) {
        shape = shapes[shape_index]
        print_timing(shape)
        print_bytes(shape)
        print_accounting(shape)
    }
}
