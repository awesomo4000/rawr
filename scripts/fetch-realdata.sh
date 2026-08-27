#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail
export LC_ALL=C

cd "$(dirname "$0")/.."

readonly upstream_commit="929d8088817840f43ffaa8592b49373b5a2d43b2"
readonly upstream_base="https://raw.githubusercontent.com/RoaringBitmap/real-roaring-datasets/${upstream_commit}"
readonly data_dir="${RAWR_REALDATA_DIR:-$PWD/misc/realdata}"

current_temp_file=""
current_temp_dir=""

cleanup() {
    if [[ -n "$current_temp_file" && -f "$current_temp_file" ]]; then
        rm -f -- "$current_temp_file"
    fi
    if [[ -n "$current_temp_dir" && -d "$current_temp_dir" ]]; then
        case "$current_temp_dir" in
            "$data_dir"/.*.extract.*) rm -rf -- "$current_temp_dir" ;;
            *) printf 'refusing to remove unexpected temporary directory: %s\n' "$current_temp_dir" >&2 ;;
        esac
    fi
}
trap cleanup EXIT INT TERM

usage() {
    printf 'usage: %s [uscensus2000] [census1881] [wikileaks-noquotes]\n' "$0" >&2
}

select_dataset() {
    case "$1" in
        uscensus2000)
            archive="uscensus2000.zip"
            expected_sha="a0f9b171883154f7675c038387fa113f7d819262c02d2f672dfbbba03b013b3d"
            expected_entries=200
            ;;
        census1881)
            archive="census1881.zip"
            expected_sha="68f4dc3a7cea6821d9cd844e027f313b5c0089c2252a3b689c0f6949e5d3c9a3"
            expected_entries=200
            ;;
        wikileaks-noquotes)
            archive="wikileaks-noquotes.zip"
            expected_sha="012d941bbd2c3fb85452233a9b82be6eb3ab4b324719425b876d30423279be99"
            expected_entries=200
            ;;
        *)
            printf 'unknown real-data dataset: %s\n' "$1" >&2
            usage
            return 2
            ;;
    esac
}

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        printf 'neither sha256sum nor shasum is available\n' >&2
        return 1
    fi
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'required command not found: %s\n' "$1" >&2
        return 1
    fi
}

datasets=("$@")
if (( ${#datasets[@]} == 0 )); then
    datasets=(uscensus2000 census1881 wikileaks-noquotes)
fi

# Reject the full request before downloading anything.
for dataset in "${datasets[@]}"; do
    select_dataset "$dataset" >/dev/null
done

mkdir -p "$data_dir"

for dataset in "${datasets[@]}"; do
    select_dataset "$dataset"
    archive_path="$data_dir/$archive"
    dataset_path="$data_dir/$dataset"

    if [[ -f "$archive_path" ]]; then
        actual_sha="$(hash_file "$archive_path")"
        if [[ "$actual_sha" != "$expected_sha" ]]; then
            printf 'archive SHA-256 mismatch: %s\nexpected: %s\nactual:   %s\n' \
                "$archive_path" "$expected_sha" "$actual_sha" >&2
            exit 1
        fi
        printf 'verified cached archive: %s\n' "$archive_path"
    else
        require_command curl
        require_command mktemp
        require_command mv
        current_temp_file="$(mktemp "$data_dir/.${archive}.download.XXXXXX")"
        printf 'downloading %s at %s\n' "$archive" "$upstream_commit"
        curl --fail --location --retry 3 --output "$current_temp_file" \
            "$upstream_base/$archive"
        actual_sha="$(hash_file "$current_temp_file")"
        if [[ "$actual_sha" != "$expected_sha" ]]; then
            printf 'downloaded archive SHA-256 mismatch: %s\nexpected: %s\nactual:   %s\n' \
                "$archive" "$expected_sha" "$actual_sha" >&2
            exit 1
        fi
        mv "$current_temp_file" "$archive_path"
        current_temp_file=""
        printf 'installed verified archive: %s\n' "$archive_path"
    fi

    if [[ -d "$dataset_path" ]]; then
        printf 'using cached extraction: %s\n' "$dataset_path"
        continue
    fi

    require_command unzip
    require_command find
    require_command mktemp
    require_command mv
    current_temp_dir="$(mktemp -d "$data_dir/.${dataset}.extract.XXXXXX")"
    unzip -q "$archive_path" -d "$current_temp_dir"
    actual_entries="$(find "$current_temp_dir" -type f | wc -l | tr -d '[:space:]')"
    if [[ "$actual_entries" != "$expected_entries" ]]; then
        printf 'archive entry-count mismatch: %s\nexpected: %s\nactual:   %s\n' \
            "$archive" "$expected_entries" "$actual_entries" >&2
        exit 1
    fi
    mv "$current_temp_dir" "$dataset_path"
    current_temp_dir=""
    printf 'installed extraction: %s\n' "$dataset_path"
done
