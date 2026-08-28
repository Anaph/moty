/* tier.c — M3: scoring LFRU + pick (libmoty-nn). */
#include "io/tier.h"

int moty_tier_pick_swap(const uint32_t *heat, int nexpert,
                          const int *pinned, int npin,
                          int *slot, int *eid, long *gain){
    if(!heat || !pinned || npin<1 || nexpert<1) return 0;
    int cold=0;
    for(int z=1;z<npin;z++) if(heat[pinned[z]]<heat[pinned[cold]]) cold=z;
    int hot=-1; uint32_t fh=0;
    for(int e=0;e<nexpert;e++){
        int resident=0;
        for(int z=0;z<npin;z++) if(pinned[z]==e){ resident=1; break; }
        if(!resident && heat[e]>fh){ fh=heat[e]; hot=e; }
    }
    if(hot<0) return 0;
    uint32_t fc=heat[pinned[cold]];
    if(fh<=fc+(fc>>2)+4) return 0;
    *slot=cold; *eid=hot; *gain=(long)fh-(long)fc;
    return 1;
}

uint64_t moty_tier_lfru_score(uint32_t heat, uint32_t last, uint32_t clock){
    uint32_t age=clock-last, recent=age<255?255-age:0;
    return ((uint64_t)heat<<8)|recent;
}

int moty_tier_pick_lfru(const uint32_t *heat, const uint32_t *last, uint32_t clock,
                          int nexpert, const int *pinned, int npin,
                          int *slot, int *eid, long *gain){
    if(!heat||!last||!pinned||npin<1||nexpert<1) return 0;
    int cold=0;
    for(int z=1;z<npin;z++)
        if(tier_lfru_score(heat[pinned[z]],last[pinned[z]],clock)<
           tier_lfru_score(heat[pinned[cold]],last[pinned[cold]],clock)) cold=z;
    int hot=-1; uint64_t hs=0;
    for(int e=0;e<nexpert;e++){
        int resident=0; for(int z=0;z<npin;z++) if(pinned[z]==e){resident=1;break;}
        uint64_t score=tier_lfru_score(heat[e],last[e],clock);
        if(!resident&&(hot<0||score>hs)){ hot=e; hs=score; }
    }
    if(hot<0) return 0;
    uint64_t cs=tier_lfru_score(heat[pinned[cold]],last[pinned[cold]],clock);
    /* Retain the existing 25%+4-frequency hysteresis in score units. */
    if(hs<=cs+(cs>>2)+(4u<<8)) return 0;
    *slot=cold; *eid=hot; *gain=(long)((hs-cs)>>8); return 1;
}

void moty_tier_decay(uint32_t *heat, int nexpert){
    for(int e=0;e<nexpert;e++) heat[e]>>=1;
}

