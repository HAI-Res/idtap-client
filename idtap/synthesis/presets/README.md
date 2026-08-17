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

Bounds contain that but do not fix it. The fix is a loss term that
follows the harmonics of the pitch the transcription says is sounding and
measures *its* decay separately from the broadband residue, so that a
model which substitutes sympathetics for the string scores worse rather
than the same. Until that exists, treat the sympathetic and chikari
levels here as an upper bound to be set by ear.

Genuinely load-bearing finds, which no amount of listening would have
produced by hand: the body resonances, the jawari depth and threshold,
and how far the loop-filter brightness had to come down.
