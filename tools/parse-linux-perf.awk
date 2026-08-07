# SPDX-License-Identifier: MPL-2.0

function bucket_for(value) {
    if (value ~ /(PageAllocator\.|mmap|munmap|madvise|mach_vm_|vm_allocate)/) return "map"
    if (value ~ /(compiler_rt\.memset|platform_memset|platform_bzero|__bzero|(^|[._])(bzero|memset))/) return "zero"
    if (value ~ /(lazyAccumulateIntoBitset|setList|lazyUnionWith|setRange|cloneContainer|appendContainer|ensureTotalCapacity|memmove|memcpy)/) return "work"
    if (value ~ /(SmpAllocator\.(alloc|free)|c_allocator_impl\.(alloc|free)|posix_memalign|(^|[._])(malloc|free)|malloc_|_xzm_|xzm_)/) return "alloc"
    return ""
}

function finish_sample(    i, bucket, leaf) {
    if (!in_sample) return
    in_sample = 0
    if (!has_wrapper) {
        delete frame
        frame_count = 0
        return
    }

    wrapper_samples++
    bucket = ""
    for (i = 1; i <= frame_count; i++) {
        bucket = bucket_for(frame[i])
        if (bucket != "") break
    }

    leaf = frame[1]
    if (bucket != "") {
        samples[bucket]++
    } else if (leaf == "" || leaf ~ /(\[unknown\]|^0x[0-9a-fA-F]+$)/) {
        unsymbolized++
        if (debug_unknown && unsymbolized <= debug_unknown) {
            print "UNKNOWN SAMPLE" > "/dev/stderr"
            for (i = 1; i <= frame_count; i++) print "  " frame[i] > "/dev/stderr"
        }
    } else {
        samples["other"]++
    }

    delete frame
    frame_count = 0
    has_wrapper = 0
}

/^[^[:space:]]/ {
    finish_sample()
    in_sample = 1
    next
}

in_sample && /^[[:space:]]+[0-9a-fA-F]+[[:space:]]/ {
    value = $0
    sub(/^[[:space:]]+[0-9a-fA-F]+[[:space:]]+/, "", value)
    sub(/\+0x[0-9a-fA-F]+[[:space:]].*$/, "", value)
    sub(/[[:space:]]+\(.*/, "", value)
    frame[++frame_count] = value
    if (value ~ /bench_croaring\.rawr_prof_timed_lazy_or/) has_wrapper = 1
    next
}

/^[[:space:]]*$/ { finish_sample() }

END {
    finish_sample()
    symbolized = samples["alloc"] + samples["map"] + samples["zero"] + samples["work"] + samples["other"]
    printf "PROFILE\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", wrapper_samples, samples["alloc"], samples["map"], samples["zero"], samples["work"], samples["other"], unsymbolized, symbolized
}
