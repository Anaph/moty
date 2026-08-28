#ifndef MOTY_TIER_H
#define MOTY_TIER_H

#include <stdint.h>

/* Pick one RAM/VRAM hot-store slot to replace from recent routing heat.
 * The fixed margin handles tiny samples; the 25% margin prevents ping-pong. */


/* LFRU: frequency is the primary signal; recency breaks close calls. A recent
 * access contributes at most 255 points while one frequency count is worth
 * 256, so a merely recent expert cannot displace a genuinely hotter one. */







/* M3: implementazioni in io/tier.c (libmoty-nn) */
int moty_tier_pick_swap(const uint32_t *heat, int nexpert,
                          const int *pinned, int npin,
                          int *slot, int *eid, long *gain);
uint64_t moty_tier_lfru_score(uint32_t heat, uint32_t last, uint32_t clock);
int moty_tier_pick_lfru(const uint32_t *heat, const uint32_t *last, uint32_t clock,
                          int nexpert, const int *pinned, int npin,
                          int *slot, int *eid, long *gain);
void moty_tier_decay(uint32_t *heat, int nexpert);

#ifndef MOTY_CORE_NO_LEGACY
#define tier_pick_swap moty_tier_pick_swap
#define tier_lfru_score moty_tier_lfru_score
#define tier_pick_lfru moty_tier_pick_lfru
#define tier_decay moty_tier_decay
#endif

#endif
