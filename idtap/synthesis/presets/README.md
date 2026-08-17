# Fitted sitar parameters

`sitar_vilayat.json` is the result of fitting the sitar model against
Vilayat Khan's alap (transcription `68002a62e0cac794f4b4a29c`), using the
transcription itself to drive the synthesis so that the comparison is
between two performances of the same notes rather than between arbitrary
audio. CMA-ES, popsize 56, 160 generations; mean band error fell from
11.7 dB to 4.0 dB.

The fit is not the last word, and the reason is worth writing down.

A long-term spectral average cannot tell where energy came from. Ring
from the played string, ring from the sympathetics, and ring from the
chikari all land in the same bands, so the optimizer is free to buy
spectral shape with whichever is cheapest — and sympathetic ring is
cheapest, because it costs nothing in attack transients. Left to itself
it produces a sitar whose played note dies early and whose taraf wash
never stops. That is what `chikari_level`, `taraf_mix` and `taraf_t60`
sitting on their ceilings mean, alongside `t60_max` down near its floor.

Capping the sympathetics shrinks that trade without removing it. Under
the current ranges the fit still puts `taraf_t60` and `taraf_drive` on
their ceilings and `taraf_damp` on its floor — as much sympathetic
energy, as bright, as it is allowed. What the cap buys is that the
chikari no longer has to be damped to pay for it: `chikari_t60` came out
at 4.18 s, which is also where listening puts it.

Ten of eighteen parameters still sit on a bound, and two of those are
known to be in the wrong direction: `chikari_level` has run to its
ceiling in three consecutive fits against a listener who calls the
chikari too loud each time, and `taraf_damp` sits at maximum brightness
against the same objection. Those two are the ones to reach for first if
the sound needs adjusting.

An earlier version of this file claimed the chikari wanted to ring for
14 s, on two grounds that both turned out to be weaker than stated. The
recording measurement behind it reads narrowband energy at the chikari
pitches, which also contains the drone, the sympathetics and the main
string playing those same raga degrees — it measures how long energy at
those pitches persists, not how long the chikari rings. And the sweep
that agreed with it was run at a hand-picked dry taraf setting; freed to
choose, the fit prefers wash-plus-short-chikari at 4.26 dB over
dry-plus-long at 4.74. A conditional optimum had been read as an
unconditional one.

The deeper fix is still outstanding: a loss term that follows the
harmonics of the pitch the transcription says is sounding and measures
*its* decay separately from the broadband residue, so a model that
substitutes sympathetics for the string scores worse rather than the
same. Capping the sympathetics denies the fit a bad option; that term
would let it tell the difference.

Genuinely load-bearing finds, which no amount of listening would have
produced by hand: the body resonances, the jawari depth and threshold,
and how far the loop-filter brightness had to come down.
