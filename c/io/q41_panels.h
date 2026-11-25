/* Lossless Group64 cache. Each four-row panel contains I/64 groups.
 * A group holds 16 bytes of scale/min, then 64 bytes of nibbles for
 * each of the two original GGUF blocks. No values are requantized. */
#ifndef MOTY_Q41_PANELS_H
#define MOTY_Q41_PANELS_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum { Q41_PANEL_ROWS = 4, Q41_PANEL_GROUP = 64, Q41_PANEL_BYTES = 144 };
/* The indexed head substitutes four uint16_t indices for the four FP16
 * pairs. Its full index table follows the last panel. */
enum { Q41_INDEXED_BYTES = 136, Q41_INDEX_CAP = 65536, Q41_INDEX_MAX_INPUT = 896 };

static inline uint64_t q41_indexed_groups_bytes(int input, int output) {
    return (uint64_t)(input/Q41_PANEL_GROUP)*(output/Q41_PANEL_ROWS)*Q41_INDEXED_BYTES;
}

static inline uint64_t q41_indexed_bytes(int input, int output) {
    return q41_indexed_groups_bytes(input,output) + Q41_INDEX_CAP*sizeof(uint32_t);
}

static inline const uint32_t *q41_indexed_table(const uint8_t *p, int input, int output) {
    return (const uint32_t *)(p+q41_indexed_groups_bytes(input,output));
}

static inline void q41_indexed_row_block(const uint8_t *group, const uint32_t *table,
                                          int r, int half, uint8_t *raw) {
    uint16_t index;
    memcpy(&index,group+2*r,sizeof(index));
    memcpy(raw,table+index,sizeof(*table));
    memcpy(raw+4,group+8+64*half+16*r,16);
}

static inline void q41_panel_row_block(const uint8_t *group, int r, int half, uint8_t *raw) {
    memcpy(raw,group+4*r,4);
    memcpy(raw+4,group+16+64*half+16*r,16);
}

#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(_WIN32)
typedef struct {
    uint64_t version, dev, ino, size, mtime, mtime_ns, ctime, ctime_ns;
    uint64_t offset, input, output, bytes, checksum;
} Q41CacheHeader;

enum { Q41_CACHE_HASH_LEGACY = 1, Q41_CACHE_HASH_PARALLEL = 2, Q41_CACHE_INDEXED = 3 };

static const char *q41_cache_suffix(uint64_t version) {
    if (version == Q41_CACHE_INDEXED) return "moty-q41-indexed";
    return version == Q41_CACHE_HASH_PARALLEL ? "moty-q41-v2" : "moty-q41";
}

static inline uint64_t q41_cache_word(uint64_t hash, const uint8_t *p) {
    uint64_t word;
    memcpy(&word,p,sizeof(word));
    return (hash ^ word)*UINT64_C(1099511628211);
}

static uint64_t q41_cache_hash(uint64_t hash, const uint8_t *p, size_t bytes) {
    for (size_t i = 0; i < bytes; i += 8) hash = q41_cache_word(hash,p+i);
    return hash;
}

typedef struct {
    uint64_t lane[4];
    unsigned next;
} Q41CacheDigest;

/* Keep four independent multiplication chains. The next lane carries
 * across write batches so the digest depends only on the payload bytes. */
static void q41_cache_digest_update(Q41CacheDigest *state, const uint8_t *p, size_t bytes) {
    size_t i = 0;
    unsigned next = state->next;
    for (; i < bytes && next; i += 8) {
        state->lane[next] = q41_cache_word(state->lane[next],p+i);
        next = (next+1)%4;
    }
    uint64_t a = state->lane[0], b = state->lane[1], c = state->lane[2], d = state->lane[3];
    for (; bytes-i >= 32; i += 32) {
        a = q41_cache_word(a,p+i);
        b = q41_cache_word(b,p+i+8);
        c = q41_cache_word(c,p+i+16);
        d = q41_cache_word(d,p+i+24);
    }
    state->lane[0] = a;
    state->lane[1] = b;
    state->lane[2] = c;
    state->lane[3] = d;
    for (; i < bytes; i += 8) {
        state->lane[next] = q41_cache_word(state->lane[next],p+i);
        next = (next+1)%4;
    }
    state->next = next;
}

static uint64_t q41_cache_digest_finish(const Q41CacheDigest *state) {
    return q41_cache_hash(0,(const uint8_t *)state->lane,sizeof(state->lane));
}

static uint64_t q41_cache_hash_payload(uint64_t version, const uint8_t *p, size_t bytes) {
    if (version == Q41_CACHE_HASH_LEGACY) return q41_cache_hash(0,p,bytes);
    Q41CacheDigest state = {0};
    q41_cache_digest_update(&state,p,bytes);
    return q41_cache_digest_finish(&state);
}

static int q41_cache_read(int fd, uint8_t *data, size_t bytes, off_t offset) {
    while (bytes) {
        ssize_t n = pread(fd,data,bytes,offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        data += n;
        offset += n;
        bytes -= (size_t)n;
    }
    return 1;
}

static int q41_cache_write(int fd, const void *data, size_t bytes) {
    const uint8_t *p = data;
    while (bytes) {
        ssize_t n = write(fd,p,bytes);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        p += n;
        bytes -= (size_t)n;
    }
    return 1;
}

typedef struct {
    uint32_t pairs[Q41_INDEX_CAP];
    unsigned slots[2*Q41_INDEX_CAP];
    unsigned count;
} Q41CacheIndex;

static int q41_cache_index_pair(Q41CacheIndex *index, const uint8_t *pair, uint8_t *dst) {
    uint32_t value;
    memcpy(&value,pair,sizeof(value));
    const unsigned mask = 2*Q41_INDEX_CAP-1;
    unsigned slot = (value*UINT32_C(2654435761)) >> 15;
    while (index->slots[slot] && index->pairs[index->slots[slot]-1] != value)
        slot = (slot+1) & mask;
    if (!index->slots[slot]) {
        if (index->count == Q41_INDEX_CAP) return 0;
        index->pairs[index->count++] = value;
        index->slots[slot] = index->count;
    }
    const uint16_t id = (uint16_t)(index->slots[slot]-1);
    memcpy(dst,&id,sizeof(id));
    return 1;
}

static int q41_cache_build(int fd, const st_tensor *t, Q41CacheHeader *h) {
    int groups = (int)(h->input/Q41_PANEL_GROUP);
    const int indexed = h->version == Q41_CACHE_INDEXED;
    const int group_bytes = indexed ? Q41_INDEXED_BYTES : Q41_PANEL_BYTES;
    const int param_bytes = indexed ? 8 : 16;
    const size_t raw_row = (size_t)groups*40;
    const size_t raw_panel = Q41_PANEL_ROWS*raw_row;
    const size_t packed_panel = (size_t)groups*group_bytes;
    const size_t panels = (size_t)(h->output/Q41_PANEL_ROWS);
    size_t batch = (128u*1024u)/(raw_panel+packed_panel);
    if (!batch) batch = 1;
    if (batch > panels) batch = panels;
    uint8_t *buf = malloc(batch*(raw_panel+packed_panel));
    if (!buf) return 0;
    Q41CacheIndex *index = indexed ? calloc(1,sizeof(*index)) : NULL;
    if (indexed && !index) { free(buf); return 0; }
    uint8_t *packed = buf+batch*raw_panel;
    uint64_t hash = 0;
    Q41CacheDigest digest = {0};
    int ok = q41_cache_write(fd,h,sizeof(*h));
    for (size_t first = 0; ok && first < panels; first += batch) {
        const size_t count = panels-first < batch ? panels-first : batch;
        ok = q41_cache_read(t->fd,buf,count*raw_panel,t->off+(off_t)(first*raw_panel));
        for (size_t panel = 0; ok && panel < count; panel++) {
            for (int g = 0; g < groups && ok; g++) {
                uint8_t *dst = packed+panel*packed_panel+g*group_bytes;
                for (int r = 0; r < Q41_PANEL_ROWS; r++) {
                    const uint8_t *src = buf+panel*raw_panel+r*raw_row+g*40;
                    if (memcmp(src,src+20,4)) { ok = 0; break; }
                    if (indexed) {
                        if (!q41_cache_index_pair(index,src,dst+2*r)) { ok = 0; break; }
                    } else memcpy(dst+4*r,src,4);
                    memcpy(dst+param_bytes+16*r,src+4,16);
                    memcpy(dst+param_bytes+64+16*r,src+24,16);
                }
            }
        }
        if (ok) {
            if (h->version == Q41_CACHE_HASH_LEGACY)
                hash = q41_cache_hash(hash,packed,count*packed_panel);
            else q41_cache_digest_update(&digest,packed,count*packed_panel);
            ok = q41_cache_write(fd,packed,count*packed_panel);
        }
    }
    free(buf);
    if (ok && indexed) {
        q41_cache_digest_update(&digest,(const uint8_t *)index->pairs,sizeof(index->pairs));
        ok = q41_cache_write(fd,index->pairs,sizeof(index->pairs));
    }
    free(index);
    h->checksum = h->version == Q41_CACHE_HASH_LEGACY ? hash : q41_cache_digest_finish(&digest);
    return ok && pwrite(fd,h,sizeof(*h),0) == sizeof(*h);
}

static int q41_cache_read_header(int fd, const Q41CacheHeader *key, Q41CacheHeader *disk,
                                 const struct stat *info) {
    return info->st_size >= 0 && (uint64_t)info->st_size == sizeof(*key)+key->bytes &&
           q41_cache_read(fd,(uint8_t *)disk,sizeof(*disk),0) &&
           !memcmp(key,disk,offsetof(Q41CacheHeader,checksum));
}

/* Open or replace a stale regular sidecar. The payload is checked by the
 * consumer. A resident consumer can request one retry after a checksum miss. */
static int st_q41_cache_open_format(shards *s, const st_tensor *t, int I, int O,
                                     Q41CacheHeader *disk, int indexed, int repair) {
    if (I <= 0 || O <= 0 || I%64 || O%4 || !g_st_map ||
        t->off < 0 || (int64_t)O*(I/64)*40 != t->nbytes) return -1;
    struct stat src;
    if (fstat(t->fd,&src) || t->off > src.st_size || t->nbytes > src.st_size-t->off) return -1;
    int shard = 0;
    while (shard < s->nfd && s->fds[shard] != t->fd) shard++;
    if (shard == s->nfd || !s->paths[shard]) return -1;
#ifdef __APPLE__
    struct timespec mt = src.st_mtimespec, ct = src.st_ctimespec;
#else
    struct timespec mt = src.st_mtim, ct = src.st_ctim;
#endif
    const char *hash = moty_getenv("MOTY_Q41_CACHE_HASH");
    int version = indexed ? Q41_CACHE_INDEXED :
        hash && !strcmp(hash,"parallel") ? Q41_CACHE_HASH_PARALLEL : Q41_CACHE_HASH_LEGACY;
    uint64_t bytes = indexed ? q41_indexed_bytes(I,O) : (uint64_t)(O/4)*(I/64)*Q41_PANEL_BYTES;
    Q41CacheHeader key = {version,src.st_dev,src.st_ino,src.st_size,mt.tv_sec,mt.tv_nsec,
                          ct.tv_sec,ct.tv_nsec,t->off,I,O,bytes,0};
    if (key.bytes > SIZE_MAX-sizeof(key)) return -1;
    char path[4096], tmp[4096];
    int n = snprintf(path,sizeof(path),"%s.%s-%lld",s->paths[shard],q41_cache_suffix(version),(long long)t->off);
    if (n < 0 || (size_t)n >= sizeof(path)-8) return -1;
    int fd = open(path,O_RDONLY | O_NONBLOCK);
    struct stat sb;
    if (fd >= 0) {
        if (fstat(fd,&sb) || !S_ISREG(sb.st_mode)) { close(fd); return -1; }
        if (q41_cache_read_header(fd,&key,disk,&sb) && !repair) return fd;
        close(fd);
    } else if (errno != ENOENT) return -1;

    snprintf(tmp,sizeof(tmp),"%s.XXXXXX",path);
    int writer = mkstemp(tmp);
    if (writer < 0) return -1;
    int ok = q41_cache_build(writer,t,&key);
    if (close(writer) != 0) ok = 0;
    if (ok && rename(tmp,path) == 0) {
        s->q41_cache_writes++;
        fd = open(path,O_RDONLY | O_NONBLOCK);
        if (fd < 0) return -1;
        if (!fstat(fd,&sb) && S_ISREG(sb.st_mode) && q41_cache_read_header(fd,&key,disk,&sb))
            return fd;
        close(fd);
    } else unlink(tmp);
    return -1;
}

static int st_q41_cache_open(shards *s, const st_tensor *t, int I, int O, Q41CacheHeader *disk) {
    return st_q41_cache_open_format(s,t,I,O,disk,0,0);
}

static const uint8_t *st_q41_cache_map(shards *s, st_tensor *t, int I, int O, int indexed) {
    if (s->nfd == ST_MAX_SHARDS) return NULL;
    for (int repair = 0; repair <= 1; repair++) {
        Q41CacheHeader disk;
        int fd = st_q41_cache_open_format(s,t,I,O,&disk,indexed,repair);
        if (fd < 0) return NULL;
        size_t size = sizeof(disk)+disk.bytes;
        void *map = mmap(NULL,size,PROT_READ,MAP_PRIVATE,fd,0);
        close(fd);
        if (map == MAP_FAILED) return NULL;
        const uint8_t *data = (const uint8_t *)map+sizeof(disk);
        if (q41_cache_hash_payload(disk.version,data,disk.bytes) == disk.checksum) {
            int slot = s->nfd++;
            s->fds[slot] = s->dfds[slot] = -1;
            s->paths[slot] = NULL;
            s->maps[slot] = map;
            s->maplen[slot] = size;
            t->used = 1;
            return data;
        }
        munmap(map,size);
    }
    return NULL;
}

static const uint8_t *st_q41_cache(shards *s, st_tensor *t, int I, int O) {
    return st_q41_cache_map(s,t,I,O,0);
}
#endif
#endif
