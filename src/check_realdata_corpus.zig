// SPDX-License-Identifier: MPL-2.0

//! No-timing checker for externally fetched real-data benchmark corpora.

const std = @import("std");
const corpus = @import("realdata_corpus.zig");

const default_root = "misc/realdata";

const Command = enum {
    list,
    check,
    check_all,
    mutation_sort,
};

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(init.gpa);
    defer args.deinit();
    _ = args.skip();

    const command_name = args.next() orelse return error.MissingCommand;
    const command = std.meta.stringToEnum(Command, command_name) orelse return error.UnknownCommand;
    switch (command) {
        .list => {
            if (args.next() != null) return error.TooManyArguments;
            for (corpus.supported_datasets) |dataset| {
                std.debug.print("DATASET\t{s}\t{d}\n", .{ dataset.name(), dataset.expectedEntries() });
            }
        },
        .check => {
            const dataset = try parseDataset(args.next());
            const root = args.next() orelse default_root;
            if (args.next() != null) return error.TooManyArguments;
            try checkDataset(init.gpa, init.io, root, dataset);
        },
        .check_all => {
            const root = args.next() orelse default_root;
            if (args.next() != null) return error.TooManyArguments;
            for (corpus.supported_datasets) |dataset| {
                try checkDataset(init.gpa, init.io, root, dataset);
            }
        },
        .mutation_sort => {
            const dataset = try parseDataset(args.next());
            const root = args.next() orelse default_root;
            if (args.next() != null) return error.TooManyArguments;

            var normal = try corpus.loadDataset(init.gpa, init.io, root, dataset);
            defer normal.deinit();
            var reversed = try corpus.loadDatasetWithReversedOrderForTesting(
                init.gpa,
                init.io,
                root,
                dataset,
            );
            defer reversed.deinit();
            if (normal.fingerprint == reversed.fingerprint) return error.OrderMutationNotDetected;
            std.debug.print(
                "MUTATION_SORT\t{s}\tnormal=0x{x:0>16}\treversed=0x{x:0>16}\n",
                .{ dataset.name(), normal.fingerprint, reversed.fingerprint },
            );
        },
    }
}

fn parseDataset(name: ?[]const u8) !corpus.Dataset {
    return corpus.Dataset.parse(name orelse return error.MissingDataset) orelse
        error.UnknownDataset;
}

fn checkDataset(
    allocator: std.mem.Allocator,
    io: std.Io,
    root: []const u8,
    dataset: corpus.Dataset,
) !void {
    var loaded = try corpus.loadDataset(allocator, io, root, dataset);
    defer loaded.deinit();
    std.debug.print(
        "CORPUS\t{s}\tentries={d}\tvalues={d}\tfingerprint=0x{x:0>16}\n",
        .{ dataset.name(), loaded.bitmaps.len, loaded.total_values, loaded.fingerprint },
    );
}
