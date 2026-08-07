# SPDX-License-Identifier: MPL-2.0

function symbol_bucket(depth,    i, value) {
    for (i = depth; i >= 0; i--) {
        value = symbol[i]
        if (value ~ /(PageAllocator\.|mmap|munmap|madvise|mach_vm_|vm_allocate)/) return "map"
        if (value ~ /(platform_memset|platform_bzero|__bzero|(^|[._])bzero)/) return "zero"
        if (value ~ /(lazyAccumulateIntoBitset|setList|lazyUnionWith|setRange|cloneContainer|appendContainer|ensureTotalCapacity|platform_memmove|platform_memcpy|(^|[._])memmove|(^|[._])memcpy)/) return "work"
        if (value ~ /(SmpAllocator\.(alloc|free)|c_allocator_impl\.(alloc|free)|posix_memalign|(^|[._])(malloc|free)|malloc_|_xzm_|xzm_)/) return "alloc"
    }
    return ""
}

function finalize(depth,    exclusive, bucket) {
    exclusive = inclusive[depth] - children[depth]
    if (exclusive < 0) exclusive = 0
    bucket = symbol_bucket(depth)
    if (bucket != "") {
        samples[bucket] += exclusive
    } else if (symbol[depth] ~ /(\?\?\?|^0x[0-9a-fA-F]+$)/) {
        unsymbolized += exclusive
    } else {
        samples["other"] += exclusive
    }
    delete inclusive[depth]
    delete children[depth]
    delete symbol[depth]
}

function parse_line(line,    plus, rest, prefix_len, count_text, name) {
    plus = index(line, "+")
    if (plus == 0) return -1
    rest = substr(line, plus + 1)
    if (match(rest, /[0-9]+/) == 0) return -1
    prefix_len = RSTART - 1
    parsed_depth = int((prefix_len - 1) / 2)
    count_text = substr(rest, RSTART, RLENGTH)
    parsed_count = count_text + 0
    name = substr(rest, RSTART + RLENGTH + 1)
    sub(/  \(in .*/, "", name)
    sub(/[[:space:]]+\+.*/, "", name)
    parsed_symbol = name
    return parsed_depth
}

/^Call graph:/ { in_graph = 1; next }
/^Total number in stack/ { in_graph = 0 }

in_graph {
    depth = parse_line($0)
    if (!inside) {
        if (depth >= 0 && parsed_symbol ~ /bench_croaring\.rawr_prof_timed_lazy_or/) {
            inside = 1
            base_depth = depth
            current_depth = 0
            inclusive[0] = parsed_count
            children[0] = 0
            symbol[0] = parsed_symbol
            wrapper_samples += parsed_count
        }
        next
    }

    if (depth < 0 || depth <= base_depth) {
        for (i = current_depth; i >= 0; i--) finalize(i)
        inside = 0
        next
    }

    relative = depth - base_depth
    for (i = current_depth; i >= relative; i--) finalize(i)
    current_depth = relative
    inclusive[relative] = parsed_count
    children[relative] = 0
    symbol[relative] = parsed_symbol
    children[relative - 1] += parsed_count
}

END {
    if (inside) for (i = current_depth; i >= 0; i--) finalize(i)
    symbolized = samples["alloc"] + samples["map"] + samples["zero"] + samples["work"] + samples["other"]
    printf "PROFILE\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", wrapper_samples, samples["alloc"], samples["map"], samples["zero"], samples["work"], samples["other"], unsymbolized, symbolized
}
