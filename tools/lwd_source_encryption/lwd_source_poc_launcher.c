// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Zero-file self-extracting launcher for the LWD source-encryption PoC.
//
// The embedded Python bundle (lwd_poc.py with lwd_source_runtime.py and
// discover_lwd_sources.py registered as in-memory pseudo-modules, plus the
// manifest and scope) and the raw 32 key bytes are compiled into this file by build_lwd_source_poc.py.  At run time
// the bundle is written to an anonymous memfd and Python is exec'd on
// /proc/self/fd/<N>; the key is base64-encoded into LWD_SOURCE_KEY_B64.  No
// cache directory, temporary file or other artifact is created.
//
// This is a delivery/obfuscation convenience, not a security boundary: the
// embedded bytes are readable by the invoking user and the key is visible in
// /proc/<pid>/environ to that same user.

#define _GNU_SOURCE
#include "lwd_source_poc_assets.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define KEY_ENV_VAR "LWD_SOURCE_KEY_B64"
#define BUNDLE_ENV_VAR "LWD_SOURCE_BUNDLED"

// Interpreter candidates probed (by actually calling execvp on each) when
// LWD_POC_PYTHON is unset.  python3 comes first because several distributions
// no longer ship a bare `python`.
static const char *const python_candidates[] = {"python3", "python"};
#define PYTHON_CANDIDATE_COUNT \
    (sizeof(python_candidates) / sizeof(python_candidates[0]))

#define EXIT_RUNTIME 1
#define EXIT_USAGE 2
#define EXIT_EXEC 127

static const char *progname = "lwd_source_poc";

// --------------------------------------------------------------------------
// Diagnostics
// --------------------------------------------------------------------------

static void fail(int code, const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "%s: ", progname);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(code);
}

// Append to a bounded buffer; returns the length used (capacity once full).
static size_t appendf(char *buffer, size_t capacity, size_t used,
                      const char *fmt, ...) {
    va_list ap;
    int written;

    if (used >= capacity) {
        return capacity;
    }
    va_start(ap, fmt);
    written = vsnprintf(buffer + used, capacity - used, fmt, ap);
    va_end(ap);
    if (written < 0) {
        return used;
    }
    if ((size_t)written >= capacity - used) {
        return capacity;
    }
    return used + (size_t)written;
}

static void usage(FILE *out) {
    fprintf(out,
            "usage: %s <subcommand> [options]\n"
            "       %s --lwd-print-meta\n"
            "       %s --lwd-extract-to DIR\n"
            "\n"
            "Zero-file aarch64 launcher for the LWD source-encryption PoC.\n"
            "The bundled Python tool runs from an anonymous memfd; arguments\n"
            "other than the launcher options below are forwarded verbatim.\n"
            "\n"
            "forwarded subcommands: discover manifest build static install "
            "launch verify uninstall\n"
            "\n"
            "launcher options:\n"
            "  --lwd-print-meta       print JSON metadata (no paths, no key)\n"
            "  --lwd-extract-to DIR   debug only: write the bundle and manifest\n"
            "  --lwd-help             show this help\n"
            "\n"
            "environment:\n"
            "  LWD_POC_PYTHON   interpreter override, disables probing\n"
            "                   (default: try python3, then python)\n"
            "  LWD_SOURCE_KEY_B64, LWD_SOURCE_BUNDLED are set by this launcher\n"
            "                   for the bundled tool; do not set them by hand\n",
            progname, progname, progname);
}

// --------------------------------------------------------------------------
// Base64 (key export and metadata helpers)
// --------------------------------------------------------------------------

static const char base64_alphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static size_t base64_encoded_length(size_t size) {
    return 4 * ((size + 2) / 3);
}

// Encode `size` bytes into `out`, without a newline or NUL terminator.  `out`
// must hold at least base64_encoded_length(size) bytes.
static void base64_encode(const unsigned char *data, size_t size, char *out) {
    size_t offset = 0;
    size_t written = 0;

    while (offset + 3 <= size) {
        uint32_t chunk = ((uint32_t)data[offset] << 16) |
                         ((uint32_t)data[offset + 1] << 8) |
                         (uint32_t)data[offset + 2];
        out[written++] = base64_alphabet[(chunk >> 18) & 0x3fu];
        out[written++] = base64_alphabet[(chunk >> 12) & 0x3fu];
        out[written++] = base64_alphabet[(chunk >> 6) & 0x3fu];
        out[written++] = base64_alphabet[chunk & 0x3fu];
        offset += 3;
    }
    if (offset < size) {
        size_t remaining = size - offset;
        uint32_t chunk = (uint32_t)data[offset] << 16;
        if (remaining == 2) {
            chunk |= (uint32_t)data[offset + 1] << 8;
        }
        out[written++] = base64_alphabet[(chunk >> 18) & 0x3fu];
        out[written++] = base64_alphabet[(chunk >> 12) & 0x3fu];
        out[written++] =
            remaining == 2 ? base64_alphabet[(chunk >> 6) & 0x3fu] : '=';
        out[written++] = '=';
    }
}

static void hex_encode(const unsigned char *in, size_t len, char *out) {
    static const char digits[] = "0123456789abcdef";
    size_t i;
    for (i = 0; i < len; i++) {
        out[i * 2] = digits[in[i] >> 4];
        out[i * 2 + 1] = digits[in[i] & 0x0f];
    }
    out[len * 2] = '\0';
}

// --------------------------------------------------------------------------
// File helpers (only used by the explicit --lwd-extract-to debug switch)
// --------------------------------------------------------------------------

static void *xmalloc(size_t size) {
    void *ptr = malloc(size == 0 ? 1 : size);
    if (ptr == NULL) {
        fail(EXIT_RUNTIME, "out of memory");
    }
    return ptr;
}

static void ensure_dir(const char *path) {
    struct stat st;
    if (mkdir(path, 0700) == 0) {
        return;
    }
    if (errno != EEXIST) {
        fail(EXIT_RUNTIME, "cannot create directory %s: %s", path,
             strerror(errno));
    }
    if (lstat(path, &st) != 0) {
        fail(EXIT_RUNTIME, "cannot stat %s: %s", path, strerror(errno));
    }
    if (!S_ISDIR(st.st_mode)) {
        fail(EXIT_RUNTIME, "%s exists and is not a directory", path);
    }
}

static void mkdir_p(const char *path) {
    char buffer[PATH_MAX];
    char *cursor;
    size_t len = strlen(path);

    if (len == 0 || len >= sizeof(buffer)) {
        fail(EXIT_RUNTIME, "invalid or too long directory path");
    }
    memcpy(buffer, path, len + 1);
    while (len > 1 && buffer[len - 1] == '/') {
        buffer[--len] = '\0';
    }
    for (cursor = buffer + 1; *cursor != '\0'; cursor++) {
        if (*cursor == '/') {
            *cursor = '\0';
            ensure_dir(buffer);
            *cursor = '/';
        }
    }
    ensure_dir(buffer);
}

static void write_all_or_fail(int fd, const unsigned char *data, size_t size,
                              const char *path) {
    size_t written = 0;
    while (written < size) {
        ssize_t chunk = write(fd, data + written, size - written);
        if (chunk < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail(EXIT_RUNTIME, "cannot write %s: %s", path, strerror(errno));
        }
        written += (size_t)chunk;
    }
}

static void write_file(const char *dir, const char *name,
                       const unsigned char *data, size_t size) {
    char path[PATH_MAX];
    int fd;

    if (snprintf(path, sizeof(path), "%s/%s", dir, name) >=
        (int)sizeof(path)) {
        fail(EXIT_RUNTIME, "output path is too long");
    }
    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
        fail(EXIT_RUNTIME, "cannot create %s: %s", path, strerror(errno));
    }
    write_all_or_fail(fd, data, size, path);
    if (close(fd) != 0) {
        fail(EXIT_RUNTIME, "cannot close %s: %s", path, strerror(errno));
    }
}

// --------------------------------------------------------------------------
// Anonymous bundle transport
// --------------------------------------------------------------------------

// Write the embedded bundle into an anonymous memfd and return the
// /proc/self/fd/<N> path Python will execute.  FD_CLOEXEC is cleared so the
// descriptor survives execvp.  Nothing touches the filesystem.
static char *bundle_memfd_path(void) {
    char path[64];
    int fd = memfd_create("lwd_source_poc", 0);

    if (fd < 0) {
        fail(EXIT_RUNTIME,
             "cannot create an anonymous memfd for the bundled script: %s",
             strerror(errno));
    }
    write_all_or_fail(fd, lwd_bundle_data, lwd_bundle_size, "memfd");
    if (fcntl(fd, F_SETFD, 0) != 0) {
        fail(EXIT_RUNTIME, "cannot clear FD_CLOEXEC on the bundle memfd: %s",
             strerror(errno));
    }
    if (snprintf(path, sizeof(path), "/proc/self/fd/%d", fd) >=
        (int)sizeof(path)) {
        fail(EXIT_RUNTIME, "memfd path is too long");
    }
    if (access(path, R_OK) != 0) {
        if (snprintf(path, sizeof(path), "/dev/fd/%d", fd) >=
            (int)sizeof(path)) {
            fail(EXIT_RUNTIME, "memfd path is too long");
        }
        if (access(path, R_OK) != 0) {
            fail(EXIT_RUNTIME,
                 "neither /proc/self/fd nor /dev/fd exposes the bundle memfd; "
                 "an accessible fd filesystem is required");
        }
    }
    return strdup(path);
}

// Export the raw key as canonical base64 plus the bundled-mode marker.  The
// encoded buffer is wiped after setenv() has copied it.
static void export_runtime_env(void) {
    size_t encoded_len = base64_encoded_length(sizeof(lwd_source_key));
    char *encoded = xmalloc(encoded_len + 1);

    base64_encode(lwd_source_key, sizeof(lwd_source_key), encoded);
    encoded[encoded_len] = '\0';
    if (setenv(KEY_ENV_VAR, encoded, 1) != 0) {
        fail(EXIT_RUNTIME, "cannot export %s: %s", KEY_ENV_VAR,
             strerror(errno));
    }
    memset(encoded, 0, encoded_len);
    free(encoded);
    if (setenv(BUNDLE_ENV_VAR, "1", 1) != 0) {
        fail(EXIT_RUNTIME, "cannot export %s: %s", BUNDLE_ENV_VAR,
             strerror(errno));
    }
}

// --------------------------------------------------------------------------
// Interpreter selection
// --------------------------------------------------------------------------

// Explicit interpreter override; NULL when LWD_POC_PYTHON is unset or empty.
static const char *explicit_python(void) {
    const char *python = getenv("LWD_POC_PYTHON");
    return (python != NULL && python[0] != '\0') ? python : NULL;
}

// Interpreter text reported by --lwd-print-meta: the override when one is set,
// otherwise the probe order the launcher will try.
static const char *python_description(void) {
    static char buffer[128];
    const char *override = explicit_python();
    size_t used = 0;
    size_t index;

    if (override != NULL) {
        return override;
    }
    buffer[0] = '\0';
    for (index = 0; index < PYTHON_CANDIDATE_COUNT; index++) {
        used = appendf(buffer, sizeof(buffer), used, "%s%s",
                       index == 0 ? "" : ", ", python_candidates[index]);
    }
    return buffer;
}

// Replace this process with the Python interpreter, keeping the memfd bundle
// path as argv[1].  An explicit LWD_POC_PYTHON is used verbatim and never falls
// back; otherwise every candidate is really exec'd in order, so a missing
// interpreter is reported by execvp itself (ENOENT/ENOTDIR/...) rather than
// guessed from PATH beforehand.  Never returns.
static void exec_python(char **python_argv) {
    char detail[512];
    const char *override = explicit_python();
    size_t used = 0;
    size_t index;

    if (override != NULL) {
        python_argv[0] = (char *)override;
        execvp(override, python_argv);
        fail(EXIT_EXEC, "cannot execute LWD_POC_PYTHON=%s: %s", override,
             strerror(errno));
    }

    detail[0] = '\0';
    for (index = 0; index < PYTHON_CANDIDATE_COUNT; index++) {
        const char *candidate = python_candidates[index];
        int saved;

        python_argv[0] = (char *)candidate;
        execvp(candidate, python_argv);
        saved = errno;
        used = appendf(detail, sizeof(detail), used, "%s%s (%s)",
                       index == 0 ? "" : ", ", candidate, strerror(saved));
    }
    fail(EXIT_EXEC,
         "cannot execute any Python interpreter; tried: %s; "
         "set LWD_POC_PYTHON to a working interpreter",
         detail);
}

// --------------------------------------------------------------------------
// Meta / extract commands
// --------------------------------------------------------------------------

static void json_print_string(const char *text) {
    const unsigned char *cursor = (const unsigned char *)text;
    putchar('"');
    for (; *cursor != '\0'; cursor++) {
        switch (*cursor) {
        case '"':
            fputs("\\\"", stdout);
            break;
        case '\\':
            fputs("\\\\", stdout);
            break;
        case '\b':
            fputs("\\b", stdout);
            break;
        case '\f':
            fputs("\\f", stdout);
            break;
        case '\n':
            fputs("\\n", stdout);
            break;
        case '\r':
            fputs("\\r", stdout);
            break;
        case '\t':
            fputs("\\t", stdout);
            break;
        default:
            if (*cursor < 0x20) {
                printf("\\u%04x", *cursor);
            } else {
                putchar(*cursor);
            }
            break;
        }
    }
    putchar('"');
}

static int cmd_print_meta(void) {
    char bundle_hex[65];
    char manifest_hex[65];
    char tool_hex[65];
    char runtime_hex[65];
    char discovery_hex[65];
    char scope_hex[65];

    hex_encode(lwd_bundle_sha256, 32, bundle_hex);
    hex_encode(lwd_manifest_sha256, 32, manifest_hex);
    hex_encode(lwd_tool_source_sha256, 32, tool_hex);
    hex_encode(lwd_runtime_source_sha256, 32, runtime_hex);
    hex_encode(lwd_discovery_source_sha256, 32, discovery_hex);
    hex_encode(lwd_scope_sha256, 32, scope_hex);

    printf("{\n");
    printf("  \"tool\": \"lwd_source_poc\",\n");
    printf("  \"format_version\": 1,\n");
    printf("  \"transport\": \"memfd\",\n");
    printf("  \"bundle_env\": \"%s=1\",\n", BUNDLE_ENV_VAR);
    printf("  \"key_source\": \"env:%s\",\n", KEY_ENV_VAR);
    printf("  \"python\": ");
    json_print_string(python_description());
    printf(",\n");
    printf("  \"assets\": [\n");
    printf("    {\"name\": \"lwd_poc.py\", \"source_bytes\": %zu, "
           "\"source_sha256\": \"%s\", \"bundle_bytes\": %zu, "
           "\"bundle_sha256\": \"%s\"},\n",
           lwd_tool_source_size, tool_hex, lwd_bundle_size, bundle_hex);
    printf("    {\"name\": \"lwd_source_runtime.py\", \"source_bytes\": %zu, "
           "\"source_sha256\": \"%s\"},\n",
           lwd_runtime_source_size, runtime_hex);
    printf("    {\"name\": \"discover_lwd_sources.py\", \"source_bytes\": %zu, "
           "\"source_sha256\": \"%s\"},\n",
           lwd_discovery_source_size, discovery_hex);
    printf("    {\"name\": \"lwd_source_scope.json\", \"bytes\": %zu, "
           "\"sha256\": \"%s\"},\n",
           lwd_scope_size, scope_hex);
    printf("    {\"name\": \"manifest.json\", \"bytes\": %zu, "
           "\"sha256\": \"%s\"}\n",
           lwd_manifest_size, manifest_hex);
    printf("  ]\n}\n");
    return 0;
}

static int cmd_extract(const char *dest) {
    mkdir_p(dest);
    write_file(dest, "lwd_poc.py", lwd_bundle_data, lwd_bundle_size);
    write_file(dest, "manifest.json", lwd_manifest_data, lwd_manifest_size);
    printf("extracted lwd_poc.py (bundle) bytes=%zu\n", lwd_bundle_size);
    printf("extracted manifest.json bytes=%zu\n", lwd_manifest_size);
    return 0;
}

// --------------------------------------------------------------------------
// Entry point
// --------------------------------------------------------------------------

int main(int argc, char **argv) {
    char *script_path;
    char **python_argv;
    int index;

    if (argc >= 2 && strcmp(argv[1], "--lwd-help") == 0) {
        usage(stdout);
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "--lwd-print-meta") == 0) {
        if (argc != 2) {
            fail(EXIT_USAGE, "--lwd-print-meta takes no further arguments");
        }
        return cmd_print_meta();
    }
    if (argc >= 2 && strcmp(argv[1], "--lwd-extract-to") == 0) {
        if (argc != 3) {
            fail(EXIT_USAGE, "usage: %s --lwd-extract-to DIR", progname);
        }
        return cmd_extract(argv[2]);
    }
    if (argc >= 2 && strncmp(argv[1], "--lwd-", 6) == 0) {
        fprintf(stderr, "%s: unknown launcher option %s\n\n", progname, argv[1]);
        usage(stderr);
        return EXIT_USAGE;
    }
    if (argc < 2) {
        usage(stderr);
        return EXIT_USAGE;
    }

    script_path = bundle_memfd_path();
    export_runtime_env();

    python_argv = xmalloc(((size_t)argc + 2) * sizeof(char *));
    python_argv[1] = script_path;
    for (index = 1; index < argc; index++) {
        python_argv[index + 1] = argv[index];
    }
    python_argv[argc + 1] = NULL;

    exec_python(python_argv);
    return EXIT_EXEC;
}
