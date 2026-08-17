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

Bounds turned out to fix more of this than expected, though not for the
reason a bound usually works.

Capping the sympathetics does not merely stop the fit overusing them; it
removes the *substitute*, and the rest of the fit reorganises around
that. With `taraf_mix` held at 0.15 the loss immediately starts
preferring a chikari that rings for 14 seconds, having previously parked
it on a 1-second floor. Fourteen seconds is also what the recording
measures directly — 12 to 18 s across the three chikari strings, by
narrowband decay at each transcribed strum. Two unrelated methods, the
same answer, from a loss that was demonstrably getting it wrong an hour
earlier.

So the sympathetic ceilings here are load-bearing rather than
conservative, and the values in this file predate them: its
`chikari_t60`, `taraf_mix`, `taraf_t60` and `taraf_damp` all now fall
outside the current ranges, and `load_preset` warns about exactly those.
The body resonances, jawari and brightness are unaffected.

The deeper fix is still outstanding: a loss term that follows the
harmonics of the pitch the transcription says is sounding and measures
*its* decay separately from the broadband residue, so a model that
substitutes sympathetics for the string scores worse rather than the
same. Capping the sympathetics is a way of denying the fit a bad option;
that term would let it tell the difference.

Genuinely load-bearing finds, which no amount of listening would have
produced by hand: the body resonances, the jawari depth and threshold,
and how far the loop-filter brightness had to come down.
