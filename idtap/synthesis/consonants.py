"""Phonetic rule data for rendering Hindi/Hindustani-vocal consonants on the
Klatt cascade/parallel formant synthesizer.

Each vocal trajectory in an IDTAP transcription may carry a start consonant
and/or an end consonant (IPA), plus a vowel.  This module supplies the
*static* per-consonant acoustic recipe; a separate integration layer turns
each recipe into control-signal gestures on the synth:

    closure -> burst -> aspiration/VOT -> formant transition into the vowel

Conventions
-----------
Frequencies
    Hz, adult male vocal tract (the vowel table in ``vowels.py`` is also a
    male table).  For a female voice the integration layer should scale
    formant loci up by roughly 1.15-1.20 (uniform scaling is crude but
    adequate here).  No frequency in this table exceeds 6500 Hz, so every
    value is comfortably below Nyquist at the 44.1 kHz render rate
    (see ``render.DEFAULT_SR``).

Levels (``frication_db``, ``bypass_db``, ``aspiration_db``)
    klatt-syn / ``kernels._db_to_lin`` convention: 0 dB = unity gain on that
    source, useful audible range roughly -30..0 dB, and -99 dB = off.
    Compare ``render.KLATT_CASCADE_ASPIRATION_DB = -25`` (the breathy
    baseline of the sustained voice), so a value like -8 dB really is a
    strongly audible aspiration burst against that background.

    NB this is *not* Klatt's own scale.  Klatt's AF/AH/AB run 0-80 dB where
    60 = "strong frication/aspiration noise" and 0 = source off.  Mapping
    into the convention used here: Klatt 60 dB -> roughly 0..-4 dB, Klatt
    ~45 dB -> roughly -15 dB, Klatt 0 -> -99.  His relative spacing between
    consonants is what has been preserved below, not his absolute numbers.

Which segment a level applies to
    * stops / affricates -- ``frication_db`` and ``burst_formant`` apply
      during ``burst_dur`` (the release burst / affricate frication).
    * fricatives -- ``burst_dur`` is 0; the frication is continuous, so
      ``frication_db`` and ``burst_formant`` apply for the whole
      ``closure_dur`` (read as "constriction duration" there).
    * ``aspiration_db`` always applies over ``aspiration_dur``, which
      follows the release (VOT lag).  For breathy-voiced segments
      (``breathy_offset``) that aspiration *overlaps* continuing voicing
      rather than replacing it.
    * ``bypass_db`` is the parallel-branch bypass path, which flattens the
      spectrum.  Klatt's Table III is unambiguous about who uses it: the
      labials are synthesized on the bypass path *alone* (/p b/ AB=63,
      /f v/ AB=57, with every parallel formant amplitude A2-A6 = 0 -- the
      flat bypass path simply is the diffuse labial spectrum), the dental
      fricatives ride mostly on bypass (AB=48), and everything else has
      AB=0 with the energy assigned to specific parallel resonators.  That
      pattern is reproduced below: bypass high for labials and the Hindi
      dental stops, low for velars, palatals, /ʃ/ and /s/.

    * Ramping (Klatt's KLSYN notes): frication should be ramped on
      gradually for fricatives -- a straight line from off to full over
      ~90 ms -- but switched on abruptly for plosive bursts.  Worth
      honouring in the integration layer; it is the difference between a
      fricative and a burst as much as the spectrum is.

``locus``
    (F1, F2, F3) target the formants are driven toward at closure and away
    from over ``transition_dur`` on release (locus theory).  ``None`` means
    "no transition shaping" -- used only for /h/, which is simply the vowel
    excited by noise.

Sources
-------
Klatt, D. H. (1980). "Software for a cascade/parallel formant synthesizer."
    JASA 67(3), 971-995.  Primary source for the synthesizer's parameter set
    (AF frication amplitude, AH aspiration amplitude, AB bypass, parallel
    burst resonators, FNP/FNZ nasal pole/zero).
    Full text: https://www.fon.hum.uva.nl/david/ma_ssp/doc/Klatt-1980-JAS000971.pdf
    Chapter-3 description of the cascade/parallel structure:
    http://languagelog.ldc.upenn.edu/myl/ldc/Turkeltaub/KlattChapter3Description.pdf
    The port this module feeds (chdh/klatt-syn) follows the same parameter
    conventions: https://github.com/chdh/klatt-syn

    His Table III (p. 987) gives parameter values for English consonants
    *before front vowels*.  The F2/F3 columns double as loci: Klatt states
    they "can serve as 'loci' for the characterization of the consonant-vowel
    formant transitions."  His values, for comparison with the loci chosen
    below:  labial 1100/2150, labiodental 1100/2080, dental fricative
    1290/2540, alveolar stop 1600/2600, alveolar fricative 1390/2530,
    postalveolar 1800-1840/2750-2820, velar 1990/2850.  Burst/frication
    assignments: labials pure bypass; /s z/ on F6 alone (A6=52, AB=0, with
    F6 fixed at 4900 Hz and B6=1000, i.e. a broad high-frequency emphasis
    rather than a narrow peak); /ʃ/ F3-dominant at 2750 (A3=57) with a
    broad F4-F6 plateau; /t d/ high-frequency rising, peaking at F5/F6;
    /k g/ compact F3 peak at 2850 (A3=53).  Klatt adds that A2 should rise
    to about 60 dB for velars before *nonfront* vowels, where F2 becomes a
    front-cavity resonance -- another reason the velar entries here are a
    compromise (see the locus notes below).

    Two further recipes taken from his text rather than the table:
    breathy voice is produced by setting aspiration equal to the voicing
    amplitude (AH = AV) -- which is exactly what ``breathy_offset`` asks the
    integration layer to do -- and /h/, absent from Table III, is made by
    "taking formant frequency and bandwidth parameters from the following
    vowel, increasing the first formant bandwidth to about 300 Hz, and
    replacing voicing by aspiration."

    Klatt 1980 contains no duration rules ("general strategies for the
    synthesis of English syllables are beyond the scope of this paper"), so
    every duration below comes from the Hindi literature cited further down,
    not from him.  His one worked timing example, [pʰa], is a 5 ms burst,
    40 ms of aspiration, and voicing onset 45 ms after the burst.

Fujimura, O. (1962). "Analysis of Nasal Consonants." JASA 34(12), 1865-1875.
    Nasal murmur structure: a very low first nasal formant near 300 Hz, high
    damping, and a place-dependent antiformant (zero) -- /m/ 750-1250 Hz,
    /n/ 1450-2200 Hz, /ŋ/ chiefly above 3000 Hz.
    https://pubs.aip.org/asa/jasa/article-abstract/34/12/1865/684899/

    The ``nasal_murmur`` column follows Fujimura, *not* Klatt.  Klatt's
    Table III gives one fixed pole/zero pair for both /m/ and /n/ (FNP=270,
    FNZ=450) and codes place through the oral formants and their bandwidths
    instead (/m/ F2/F3 = 1270/2130 with B2=B3=200; /n/ 1340/2470 with
    B2=B3=300), explaining that "the details of nasal murmurs described by
    Fujimura (1962) are approximated by formant bandwidth adjustments rather
    than by the theoretically correct method of pole-zero insertion" --
    because sliding the zero at release would click.  Offline rendering has
    no such constraint: we can afford the correct method, so the per-place
    zeros are given here and the integration layer is free to cross-fade
    them.  Klatt has no /ŋ/ at all.  His nasal defaults for a *non*-nasal
    segment are FNP = FNZ = 250 Hz, BNP = BNZ = 100 (pole and zero cancel).

    Values as summarised in J. Coleman, "Acoustic structure of consonants":
    https://www.phon.ox.ac.uk/jcoleman/consonant_acoustics.htm
    (same page: bilabial bursts 600-800 Hz; velar bursts compact near
    1800-2000 Hz rising to ~4700 before front vowels; alveolar bursts
    diffuse with energy above 4000 Hz; /s/ peaks near 4500 and 7500 Hz;
    /ʃ/ peak near 4000 Hz with noise from ~2000 Hz.)

Delattre, Liberman & Cooper (1955), "Acoustic loci and transitional cues for
    consonants." JASA 27, 769-773.  F2 locus theory: labial ~700-800 Hz,
    alveolar/dental ~1700-1800 Hz, velar strongly vowel-dependent.

Stevens, K. N. (1998). "Acoustic Phonetics" (MIT Press), and
Stevens & Blumstein (1975), "Quantal aspects of consonant production and
    perception."  Retroflexion lowers F3 (and F4) because of the sublingual
    cavity; compact vs. diffuse burst spectra by place.

Retroflex acoustics, Indo-Aryan specifics:
    Hamann, S. (2003), "The Phonetics and Phonology of Retroflexes," ch. 3
    "Acoustic cues and perceptual properties of retroflexes" --
    https://dspace.library.uu.nl/bitstream/handle/1874/627/c3.pdf
    Comparative acoustic-phonetic analysis of retroflex consonants of some
    Indian languages, ICA 2019 (Hindi/Marathi/Nepali): the significant cues
    are burst F2/F3/F4 plus adjoining transitions, "rising of F2 and
    lowering of F3 and F4" -- https://pub.dega-akustik.de/ICA2019/data/articles/000954.pdf

Hindi VOT (four-way stop contrast):
    Lisker, L. & Abramson, A. S. (1964), "A Cross-Language Study of Voicing
    in Initial Stops: Acoustical Measurements." Word 20, 384-422.
    https://www.tandfonline.com/doi/pdf/10.1080/00437956.1964.11659830
    Their Table 10 (p. 398), isolated words, one speaker, ms:

        place       vl.unasp   vl.asp   voiced    voiced asp.
        labial        +13       +70      -85         -61
        dental        +15       +67      -87         -87
        retroflex      +9       +60      -76         -77
        velar         +18       +92      -63         -75

    (cross-checked against the same table as reproduced in Kansas Working
    Papers in Linguistics 33 (2012), p. 29 --
    https://pdfs.semanticscholar.org/e5aa/09cfe728846458d7a8b45137f7a4cbf73295.pdf)
    The negative numbers are voice *lead*: Hindi voiced stops are prevoiced,
    which is what ``voice_bar`` asks for.  Crucially, L&A give the breathy
    release no duration at all -- they report only lead, note that /g/ and
    /gʱ/ leads "occupy ranges that are nearly coextensive," and conclude
    Hindi is a case "in which the measure of voice onset time is
    insufficient."  Their description of the murmur -- "a period of glottal
    periodicity, sometimes intermittent, mingled with random noise in the
    formant regions, all at relatively low amplitude" -- is simultaneous
    voicing plus noise, i.e. Klatt's AH = AV.

    Patil, V. & Rao, P. (2016). "Detection of phonemic aspiration for spoken
    Hindi pronunciation evaluation." J. Phonetics 54, 202-221.
    https://www.ee.iitb.ac.in/course/~daplab/publications/2016/VP_PR_Phonetics_2016.pdf
    Twenty speakers rather than L&A's one, and -- the reason it matters here
    -- it measures the voiced stops' *lag*, i.e. the murmured release L&A
    declined to time.  Means, ms (SD in the paper):

        voiceless VOT   dental 25/74   retroflex 19/65   palatal 55/90   velar 40/90
        voiced lag      labial 12/26   dental 20/34      retroflex 12/24
                        palatal 64/99  velar 38/59            (plain/breathy)

    Release timing below is set so that ``burst_dur + aspiration_dur``
    reproduces these means, with the voiceless-unaspirated values blended
    between the two studies (L&A's single speaker runs short relative to
    later Indic work).  The palatal numbers are large because they include
    the affricate frication, which this schema keeps in ``burst_dur``.

    Mikuteit, S. & Reetz, H. (2007), "Caught in the ACT: The Timing of
    Aspiration and Voicing in East Bengali." Language and Speech 50, 247-277.
    https://journals.sagepub.com/doi/abs/10.1177/00238309070500020401
    Source for the closure-duration ordering used below: labial, dental and
    retroflex stops group together with longer closures, velars are shorter.

Note on modelling limits
------------------------
The Klatt voice kernel currently rendered by ``kernels.klatt_voice`` runs
the cascade branch with the nasal pole/zero disabled and no frication
source.  The fields here therefore describe the *target* gesture; the
integration layer is expected to enable those paths (or approximate them,
e.g. shaping a noise burst through a biquad instead of a true parallel
branch) as it grows.  Fields it cannot yet honour can simply be ignored
without invalidating the rest of the entry.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

__all__ = ['ConsonantSpec', 'CONSONANTS', 'DEFAULT_SPEC', 'ALIASES',
           'get_consonant']


@dataclass(frozen=True)
class ConsonantSpec:
    ipa: str
    place: str            # 'velar'|'palatal'|'retroflex'|'dental'|'labial'|'labiodental'|'postalveolar'|'glottal'
    manner: str           # 'stop'|'nasal'|'fricative'|'affricate'|'approximant'|'trill'|'lateral'
    voiced: bool
    aspirated: bool       # True for ʰ/ʱ series (breathy-voiced for voiced aspirates)
    locus: Optional[Tuple[float, float, float]]      # F1,F2,F3 locus in Hz (adult male), None = no transition shaping
    transition_dur: float  # seconds, locus->vowel formant transition (typ. 0.04-0.08)
    closure_dur: float     # seconds of oral closure/constriction (typ. 0.05-0.09)
    burst_dur: float       # seconds of release burst frication (0 for non-stops)
    burst_formant: Optional[Tuple[float, float]]     # (freq Hz, bw Hz) of parallel frication resonance for burst/frication
    frication_db: float    # frication source level during burst/frication segment (dB; -99 = off)
    bypass_db: float       # parallel bypass level during frication (diffuse spectra ride higher)
    aspiration_dur: float  # seconds of aspiration/VOT after release
    aspiration_db: float   # aspiration source level during that phase
    nasal_murmur: Optional[Tuple[float, float, float, float]]  # (nasal formant f, bw, antiformant f, bw) during closure
    voice_bar: bool        # voiced stops: low-freq voicing during closure (Hindi voiced stops are prevoiced)
    breathy_offset: bool   # voiced aspirates (ʱ): breathy-voiced release overlapping voicing


# ---------------------------------------------------------------------------
# Locus reference values used throughout (adult male)
# ---------------------------------------------------------------------------
# Where Delattre and Klatt disagree, Delattre is followed for F2 and Klatt
# for F3.  Klatt's Table III F2 loci are measured before *front* vowels and
# so run high (labial 1100 where Delattre's classic bilabial locus is
# 700-800); his F3 values have no such bias and are adopted directly, which
# is why the labial series below carries F3 = 2150, exactly his figure.
#
# labial      F2 ~ 750    (Delattre et al. 1955: bilabial locus 700-800 Hz)
# dental      F2 ~ 1750, F3 ~ 2700   (Hindi dentals sit slightly below the
#                                     English alveolar locus of ~1800 Hz)
# retroflex   F2 ~ 1650, F3 ~ 2000   (F3 lowered ~600-700 Hz vs. the dental
#                                     series -- the defining retroflex cue;
#                                     F4 lowers too but the synth's upper
#                                     formants are fixed, so F3 carries it)
# palatal     F2 ~ 1900, F3 ~ 2700
# velar       F2 ~ 2000, F3 ~ 2450
#   NOTE on velars: the velar F2 locus is the most vowel-dependent of all --
#   roughly 3000 Hz before front vowels and ~1300 Hz before back vowels, with
#   the classic "velar pinch" of F2 and F3 converging at release.  We cannot
#   condition on the following vowel in this table (the integration layer
#   gets the vowel and may refine it), so a single compromise value near
#   2000 Hz is used, with F3 pulled down to 2450 to keep the pinch visible.
# glottal /h/ has no locus at all: it is the following vowel's own formants
#   excited by noise.

CONSONANTS: Dict[str, ConsonantSpec] = {

    # -- ka-varga (velar) ---------------------------------------------------

    # क  ka  — "kalyāṇ" (rāg); tabla bol "ke/kat".
    # VOT +18 ms (L&A) / +40 ms (Patil & Rao); blended to ~30 ms.
    # Throughout this table VOT is burst_dur + aspiration_dur, not either alone.
    'k': ConsonantSpec(
        ipa='k', place='velar', manner='stop', voiced=False, aspirated=False,
        locus=(250.0, 2000.0, 2450.0), transition_dur=0.050,
        closure_dur=0.060,          # velar closures are the shortest (Mikuteit & Reetz 2007)
        burst_dur=0.012,            # velar release is long/multi-burst
        burst_formant=(2200.0, 400.0),   # compact mid burst, narrow bw (Klatt /k/: A3 at 2850)
        frication_db=-6.0, bypass_db=-22.0,
        aspiration_dur=0.018, aspiration_db=-16.0,
        nasal_murmur=None, voice_bar=False, breathy_offset=False),

    # ख  kha — "kharaj" (the low Sa); tabla bol "kha".
    # VOT +92 ms (L&A) and +90 ms (Patil & Rao) — the two studies agree.
    'kʰ': ConsonantSpec(
        ipa='kʰ', place='velar', manner='stop', voiced=False, aspirated=True,
        locus=(250.0, 2000.0, 2450.0), transition_dur=0.050,
        closure_dur=0.060, burst_dur=0.012,
        burst_formant=(2200.0, 400.0),
        frication_db=-6.0, bypass_db=-22.0,
        aspiration_dur=0.078, aspiration_db=-8.0,
        nasal_murmur=None, voice_bar=False, breathy_offset=False),

    # ग  ga  — "gandhār" (Ga); tabla bol "ge/ghe".
    # Prevoiced (lead -63 ms, L&A); release lag 38 ms (Patil & Rao).
    'g': ConsonantSpec(
        ipa='g', place='velar', manner='stop', voiced=True, aspirated=False,
        locus=(250.0, 2000.0, 2450.0), transition_dur=0.050,
        closure_dur=0.065, burst_dur=0.012,
        burst_formant=(2000.0, 500.0),   # voiced bursts weaker and less sharp
        frication_db=-14.0, bypass_db=-26.0,
        aspiration_dur=0.026, aspiration_db=-30.0,
        nasal_murmur=None, voice_bar=True, breathy_offset=False),

    # घ  gha — "gharānā", "ghar".  Prevoiced (-75 ms, L&A) with a 59 ms
    # murmured release (Patil & Rao) — the longest of the voiced aspirates.
    'gʱ': ConsonantSpec(
        ipa='gʱ', place='velar', manner='stop', voiced=True, aspirated=True,
        locus=(250.0, 2000.0, 2450.0), transition_dur=0.050,
        closure_dur=0.065, burst_dur=0.012,
        burst_formant=(2000.0, 500.0),
        frication_db=-14.0, bypass_db=-26.0,
        aspiration_dur=0.047, aspiration_db=-14.0,   # murmur, not silence
        nasal_murmur=None, voice_bar=True, breathy_offset=True),

    # ङ  ṅa  — velar nasal, only in clusters/finals: "raṅg", "aṅg".
    'ŋ': ConsonantSpec(
        ipa='ŋ', place='velar', manner='nasal', voiced=True, aspirated=False,
        locus=(250.0, 2100.0, 2400.0), transition_dur=0.050,
        closure_dur=0.070, burst_dur=0.0, burst_formant=None,
        frication_db=-99.0, bypass_db=-99.0,
        aspiration_dur=0.0, aspiration_db=-99.0,
        # Fujimura 1962: /ŋ/'s main antiformant lies above 3000 Hz, i.e. it
        # barely shapes the murmur -- placed high and wide so its effect is
        # weak rather than absent.
        nasal_murmur=(280.0, 90.0, 3800.0, 400.0),
        voice_bar=True, breathy_offset=False),

    # -- ca-varga (palatal; phonetically postalveolar affricates in Hindi) ---

    # च  ca  — "candra", "cīz".  Realised [tʃ]: stop closure + long frication.
    'c': ConsonantSpec(
        ipa='c', place='palatal', manner='affricate', voiced=False, aspirated=False,
        locus=(250.0, 1900.0, 2700.0), transition_dur=0.050,
        closure_dur=0.055,
        burst_dur=0.050,            # affricate frication, not a stop burst
        burst_formant=(3000.0, 400.0),   # postalveolar, compact (Klatt /tʃ/: F3 2820)
        frication_db=-6.0, bypass_db=-20.0,
        # Patil & Rao's 55 ms palatal "VOT" spans frication + lag together.
        aspiration_dur=0.005, aspiration_db=-16.0,
        nasal_murmur=None, voice_bar=False, breathy_offset=False),

    # छ  cha — "chand", "chōṭā (khayāl)".
    'cʰ': ConsonantSpec(
        ipa='cʰ', place='palatal', manner='affricate', voiced=False, aspirated=True,
        locus=(250.0, 1900.0, 2700.0), transition_dur=0.050,
        closure_dur=0.055, burst_dur=0.050,
        burst_formant=(3000.0, 400.0),
        frication_db=-6.0, bypass_db=-20.0,
        aspiration_dur=0.040, aspiration_db=-8.0,   # frication + lag ~ 90 ms (P&R)
        nasal_murmur=None, voice_bar=False, breathy_offset=False),

    # ज  ja  — "jaltaraṅg", "jōṛ".  Realised [dʒ], prevoiced.
    'ɟ': ConsonantSpec(
        ipa='ɟ', place='palatal', manner='affricate', voiced=True, aspirated=False,
        locus=(250.0, 1900.0, 2700.0), transition_dur=0.050,
        closure_dur=0.060, burst_dur=0.055,
        burst_formant=(2800.0, 450.0),
        frication_db=-14.0, bypass_db=-24.0,
        aspiration_dur=0.009, aspiration_db=-30.0,   # voiced lag ~ 64 ms (P&R)
        nasal_murmur=None, voice_bar=True, breathy_offset=False),

    # झ  jha — "jhālā", "jhaptāl".  Breathy lag 99 ms (P&R) — the longest
    # release in the inventory, frication and murmur together.
    'ɟʱ': ConsonantSpec(
        ipa='ɟʱ', place='palatal', manner='affricate', voiced=True, aspirated=True,
        locus=(250.0, 1900.0, 2700.0), transition_dur=0.050,
        closure_dur=0.060, burst_dur=0.055,
        burst_formant=(2800.0, 450.0),
        frication_db=-14.0, bypass_db=-24.0,
        aspiration_dur=0.044, aspiration_db=-14.0,
        nasal_murmur=None, voice_bar=True, breathy_offset=True),

    # ञ  ña  — palatal nasal, in clusters: "pañcam" (Pa).
    'ɲ': ConsonantSpec(
        ipa='ɲ', place='palatal', manner='nasal', voiced=True, aspirated=False,
        locus=(250.0, 2200.0, 2800.0), transition_dur=0.050,
        closure_dur=0.070, burst_dur=0.0, burst_formant=None,
        frication_db=-99.0, bypass_db=-99.0,
        aspiration_dur=0.0, aspiration_db=-99.0,
        nasal_murmur=(280.0, 90.0, 3200.0, 350.0),   # zero above /n/, below /ŋ/
        voice_bar=True, breathy_offset=False),

    # -- ṭa-varga (retroflex): defining cue is a strongly lowered F3 ---------

    # ट  ṭa  — "ṭukṛā", "ṭappā".  VOT +9 ms (L&A) / +19 ms (P&R) -> ~15 ms.
    'ʈ': ConsonantSpec(
        ipa='ʈ', place='retroflex', manner='stop', voiced=False, aspirated=False,
        locus=(250.0, 1650.0, 2000.0),   # F3 ~700 Hz below the dental series
        transition_dur=0.055,            # longer: F3 has far to travel
        closure_dur=0.075, burst_dur=0.010,
        burst_formant=(2600.0, 600.0),   # burst lower/more compact than dental
        frication_db=-8.0, bypass_db=-18.0,
        aspiration_dur=0.005, aspiration_db=-16.0,
        nasal_murmur=None, voice_bar=False, breathy_offset=False),

    # ठ  ṭha — "ṭhumrī", "ṭhāṭ", "ṭhēkā".
    'ʈʰ': ConsonantSpec(
        ipa='ʈʰ', place='retroflex', manner='stop', voiced=False, aspirated=True,
        locus=(250.0, 1650.0, 2000.0), transition_dur=0.055,
        closure_dur=0.075, burst_dur=0.010,
        burst_formant=(2600.0, 600.0),
        frication_db=-8.0, bypass_db=-18.0,
        aspiration_dur=0.055, aspiration_db=-8.0,   # L&A +60, P&R +65
        nasal_murmur=None, voice_bar=False, breathy_offset=False),

    # ड  ḍa  — "ḍaggā" (bāyāṅ drum), "ḍamarū".  Lead -76 ms, lag 12 ms.
    'ɖ': ConsonantSpec(
        ipa='ɖ', place='retroflex', manner='stop', voiced=True, aspirated=False,
        locus=(250.0, 1650.0, 2000.0), transition_dur=0.055,
        closure_dur=0.075, burst_dur=0.007,
        burst_formant=(2400.0, 650.0),
        frication_db=-16.0, bypass_db=-24.0,
        aspiration_dur=0.005, aspiration_db=-30.0,
        nasal_murmur=None, voice_bar=True, breathy_offset=False),

    # ढ  ḍha — "ḍhōlak", "ḍhaiyā".  Breathy lag 24 ms (P&R).
    'ɖʱ': ConsonantSpec(
        ipa='ɖʱ', place='retroflex', manner='stop', voiced=True, aspirated=True,
        locus=(250.0, 1650.0, 2000.0), transition_dur=0.055,
        closure_dur=0.075, burst_dur=0.007,
        burst_formant=(2400.0, 650.0),
        frication_db=-16.0, bypass_db=-24.0,
        aspiration_dur=0.017, aspiration_db=-14.0,
        nasal_murmur=None, voice_bar=True, breathy_offset=True),

    # न  na  — dental/alveolar nasal: "nā", "tin", "sanam".
    # This slot is varga position 15, where strict ordering would put ṇa, but
    # IDTAP's own consonant vocabulary labels it न / "na" / example "no" — the
    # same label it gives 'n̪' — so transcribers picking it mean a plain dental
    # n, and it is the second most common consonant in the corpus. Rendered
    # dentally for that reason; the true retroflex ṇ is available as 'ɳ'.
    'n': ConsonantSpec(
        ipa='n', place='dental', manner='nasal', voiced=True, aspirated=False,
        locus=(250.0, 1650.0, 2700.0), transition_dur=0.055,
        closure_dur=0.072, burst_dur=0.0, burst_formant=None,
        frication_db=-99.0, bypass_db=-99.0,
        aspiration_dur=0.0, aspiration_db=-99.0,
        nasal_murmur=(280.0, 90.0, 1700.0, 250.0),
        voice_bar=True, breathy_offset=False),

    # -- ta-varga (dental) --------------------------------------------------

    # त  ta  — dental; tabla bols "tā", "tin", "tirakiṭa".
    # VOT +15 ms (L&A) / +25 ms (P&R) -> ~20 ms.
    # Burst: Klatt's English alveolar /t/ is high-frequency rising with all
    # its energy at F5/F6 and no bypass, but Hindi's laminal *dental* burst
    # is lower and flatter than that, so the resonance is dropped to 3500 Hz
    # and part of the energy is moved onto the bypass path.
    't': ConsonantSpec(
        ipa='t', place='dental', manner='stop', voiced=False, aspirated=False,
        locus=(250.0, 1750.0, 2700.0), transition_dur=0.045,
        closure_dur=0.070, burst_dur=0.008,
        burst_formant=(3500.0, 800.0),   # diffuse-rising; wide bw + high bypass
        frication_db=-8.0, bypass_db=-12.0,
        aspiration_dur=0.012, aspiration_db=-16.0,
        nasal_murmur=None, voice_bar=False, breathy_offset=False),

    # थ  tha — "thāṭ", "thāp".
    'tʰ': ConsonantSpec(
        ipa='tʰ', place='dental', manner='stop', voiced=False, aspirated=True,
        locus=(250.0, 1750.0, 2700.0), transition_dur=0.045,
        closure_dur=0.070, burst_dur=0.008,
        burst_formant=(3500.0, 800.0),
        frication_db=-8.0, bypass_db=-12.0,
        aspiration_dur=0.064, aspiration_db=-8.0,   # L&A +67, P&R +74
        nasal_murmur=None, voice_bar=False, breathy_offset=False),

    # द  da  — "dādrā", "drut"; tabla bol "din".  Lead -87 ms, lag 20 ms.
    'd': ConsonantSpec(
        ipa='d', place='dental', manner='stop', voiced=True, aspirated=False,
        locus=(250.0, 1750.0, 2700.0), transition_dur=0.045,
        closure_dur=0.070, burst_dur=0.006,
        burst_formant=(3200.0, 800.0),
        frication_db=-16.0, bypass_db=-18.0,
        aspiration_dur=0.014, aspiration_db=-30.0,
        nasal_murmur=None, voice_bar=True, breathy_offset=False),

    # ध  dha — "dhamār", "dhrupad"; tabla bols "dhā", "dhin".
    # Breathy lag 34 ms (P&R), the longest of the non-velar voiced aspirates.
    'dʱ': ConsonantSpec(
        ipa='dʱ', place='dental', manner='stop', voiced=True, aspirated=True,
        locus=(250.0, 1750.0, 2700.0), transition_dur=0.045,
        closure_dur=0.070, burst_dur=0.006,
        burst_formant=(3200.0, 800.0),
        frication_db=-16.0, bypass_db=-18.0,
        aspiration_dur=0.028, aspiration_db=-14.0,
        nasal_murmur=None, voice_bar=True, breathy_offset=True),

    # न  na  — dental nasal: "niṣād" (Ni), "nōm-tōm"; tabla bol "nā".
    'n̪': ConsonantSpec(
        ipa='n̪', place='dental', manner='nasal', voiced=True, aspirated=False,
        locus=(250.0, 1650.0, 2700.0), transition_dur=0.050,
        closure_dur=0.070, burst_dur=0.0, burst_formant=None,
        frication_db=-99.0, bypass_db=-99.0,
        aspiration_dur=0.0, aspiration_db=-99.0,
        nasal_murmur=(280.0, 90.0, 1700.0, 250.0),   # Fujimura /n/: 1450-2200
        voice_bar=True, breathy_offset=False),

    # -- pa-varga (labial) --------------------------------------------------

    # प  pa  — "pañcam" (Pa), "pakaṛ".  VOT +13 ms (L&A; P&R have no labial).
    # Klatt synthesizes /p b/ on the bypass path alone (AB=63, A2-A6 all 0):
    # the flat bypass *is* the diffuse labial burst.  Hence bypass_db here
    # sits above frication_db, the only place in the table where it does.
    'p': ConsonantSpec(
        ipa='p', place='labial', manner='stop', voiced=False, aspirated=False,
        locus=(250.0, 750.0, 2150.0), transition_dur=0.050,
        closure_dur=0.085,          # labial closures the longest
        burst_dur=0.006,            # and the release the briefest
        burst_formant=(800.0, 800.0),    # weak, low, diffuse-falling
        frication_db=-14.0, bypass_db=-12.0,   # diffuse -> bypass carries it
        aspiration_dur=0.007, aspiration_db=-18.0,   # L&A +13 ms
        nasal_murmur=None, voice_bar=False, breathy_offset=False),

    # फ  pha — "phir"; often realised [f] by many speakers (see ALIASES).
    'pʰ': ConsonantSpec(
        ipa='pʰ', place='labial', manner='stop', voiced=False, aspirated=True,
        locus=(250.0, 750.0, 2150.0), transition_dur=0.050,
        closure_dur=0.085, burst_dur=0.008,
        burst_formant=(800.0, 900.0),
        frication_db=-14.0, bypass_db=-12.0,
        aspiration_dur=0.062, aspiration_db=-8.0,   # L&A +70 ms
        nasal_murmur=None, voice_bar=False, breathy_offset=False),

    # ब  ba  — "bandiś", "bōl", "baṛā (khayāl)".  Lead -85 ms, lag 12 ms.
    'b': ConsonantSpec(
        ipa='b', place='labial', manner='stop', voiced=True, aspirated=False,
        locus=(250.0, 750.0, 2150.0), transition_dur=0.050,
        closure_dur=0.085, burst_dur=0.005,
        burst_formant=(700.0, 900.0),
        frication_db=-22.0, bypass_db=-20.0,
        aspiration_dur=0.007, aspiration_db=-30.0,
        nasal_murmur=None, voice_bar=True, breathy_offset=False),

    # भ  bha — "bhairav", "bhajan".  Breathy lag 26 ms (P&R).
    'bʱ': ConsonantSpec(
        ipa='bʱ', place='labial', manner='stop', voiced=True, aspirated=True,
        locus=(250.0, 750.0, 2150.0), transition_dur=0.050,
        closure_dur=0.085, burst_dur=0.005,
        burst_formant=(700.0, 900.0),
        frication_db=-22.0, bypass_db=-20.0,
        aspiration_dur=0.021, aspiration_db=-14.0,
        nasal_murmur=None, voice_bar=True, breathy_offset=True),

    # म  ma  — "madhyam" (Ma), "mīṇḍ"; extremely common in sargam/ākār text.
    'm': ConsonantSpec(
        ipa='m', place='labial', manner='nasal', voiced=True, aspirated=False,
        locus=(250.0, 800.0, 2150.0), transition_dur=0.050,
        closure_dur=0.075, burst_dur=0.0, burst_formant=None,
        frication_db=-99.0, bypass_db=-99.0,
        aspiration_dur=0.0, aspiration_db=-99.0,
        nasal_murmur=(280.0, 90.0, 1000.0, 200.0),   # Fujimura /m/: 750-1250
        voice_bar=True, breathy_offset=False),

    # -- antaḥstha (semivowels) and ūṣma (sibilants) + h --------------------

    # य  ya  — "yaman" (rāg), "yah".  High front glide.
    'j': ConsonantSpec(
        ipa='j', place='palatal', manner='approximant', voiced=True, aspirated=False,
        locus=(250.0, 2300.0, 3000.0),
        transition_dur=0.075,       # glides transition slowly by definition
        closure_dur=0.045, burst_dur=0.0, burst_formant=None,
        frication_db=-99.0, bypass_db=-99.0,
        aspiration_dur=0.0, aspiration_db=-99.0,
        nasal_murmur=None, voice_bar=True, breathy_offset=False),

    # र  ra  — "rāg", "rasa"; tabla bol "re".  Hindi /r/ is an alveolar
    # tap [ɾ]: very short "closure", fast transitions.  F3 is only mildly
    # lowered -- nothing like the American English [ɹ] (~1600-1800 Hz) -- so
    # 2000 Hz sits at the conservative edge of a lowered-F3 treatment.
    'r': ConsonantSpec(
        ipa='r', place='dental', manner='trill', voiced=True, aspirated=False,
        locus=(300.0, 1500.0, 2000.0), transition_dur=0.030,
        closure_dur=0.020,          # tap-length occlusion
        burst_dur=0.0, burst_formant=None,
        frication_db=-99.0, bypass_db=-99.0,
        aspiration_dur=0.0, aspiration_db=-99.0,
        nasal_murmur=None, voice_bar=True, breathy_offset=False),

    # ल  la  — "lay" (tempo), "lalit" (rāg).
    # Lateral: low F2, F3 near neutral.  A real lateral also has a zero
    # around 2-3 kHz from the side channels; not modelled here (no
    # antiformant outside the nasal path), so F3 is left at 2600.
    'l': ConsonantSpec(
        ipa='l', place='dental', manner='lateral', voiced=True, aspirated=False,
        locus=(350.0, 1300.0, 2600.0), transition_dur=0.060,
        closure_dur=0.050, burst_dur=0.0, burst_formant=None,
        frication_db=-99.0, bypass_db=-99.0,
        aspiration_dur=0.0, aspiration_db=-99.0,
        nasal_murmur=None, voice_bar=True, breathy_offset=False),

    # व  va  — "vādī", "vilambit".  Labiodental approximant [ʋ] in Hindi,
    # not a fricative: no frication source, just a low F2 target.
    'v': ConsonantSpec(
        ipa='v', place='labiodental', manner='approximant', voiced=True, aspirated=False,
        locus=(350.0, 1000.0, 2300.0), transition_dur=0.070,
        closure_dur=0.050, burst_dur=0.0, burst_formant=None,
        frication_db=-99.0, bypass_db=-99.0,
        aspiration_dur=0.0, aspiration_db=-99.0,
        nasal_murmur=None, voice_bar=True, breathy_offset=False),

    # श  śa  — "śuddha", "śrī" (rāg).  Compact postalveolar noise.
    # Fricatives: burst_dur is 0 and the frication runs for closure_dur.
    'ʃ': ConsonantSpec(
        ipa='ʃ', place='postalveolar', manner='fricative', voiced=False, aspirated=False,
        locus=(300.0, 1800.0, 2600.0), transition_dur=0.050,
        closure_dur=0.110, burst_dur=0.0,
        burst_formant=(3000.0, 400.0),   # compact peak; narrow bw
        frication_db=-4.0, bypass_db=-20.0,
        aspiration_dur=0.0, aspiration_db=-99.0,
        nasal_murmur=None, voice_bar=False, breathy_offset=False),

    # ष  ṣa  — "ṣaḍja" (Sa), "niṣād" (Ni).  Retroflex sibilant; in modern
    # spoken Hindi it is usually merged with श, so this is deliberately close
    # to /ʃ/ -- distinguished by a lower, flatter peak and a lowered F3 locus.
    'ʂ': ConsonantSpec(
        ipa='ʂ', place='retroflex', manner='fricative', voiced=False, aspirated=False,
        locus=(300.0, 1700.0, 2050.0), transition_dur=0.055,
        closure_dur=0.105, burst_dur=0.0,
        burst_formant=(2600.0, 550.0),   # lower and flatter than /ʃ/
        frication_db=-5.0, bypass_db=-18.0,
        aspiration_dur=0.0, aspiration_db=-99.0,
        nasal_murmur=None, voice_bar=False, breathy_offset=False),

    # स  sa  — "sargam", "sthāyī"; the sargam syllable "sā" itself.
    # Diffuse high-frequency noise, energy from ~2 kHz to above 8 kHz with
    # peaks near 4.5 and 7.5 kHz.  Klatt puts /s/ entirely on his highest
    # parallel resonator and uses *no* bypass at all (A6=52, AB=0, F6=4900,
    # B6=1000) -- the width of that resonator, not a flat path, is what makes
    # the spectrum diffuse.  Followed here: one broad resonance (bw 1000 Hz,
    # matching his B6) placed at 6000 Hz to sit between the two observed
    # peaks, with the bypass well down.  6000 Hz is safely below Nyquist at
    # 44.1 kHz even allowing for the skirt of that wide resonance.
    's': ConsonantSpec(
        ipa='s', place='dental', manner='fricative', voiced=False, aspirated=False,
        locus=(280.0, 1750.0, 2650.0), transition_dur=0.050,
        closure_dur=0.115, burst_dur=0.0,
        burst_formant=(6000.0, 1000.0),
        frication_db=-4.0, bypass_db=-16.0,
        aspiration_dur=0.0, aspiration_db=-99.0,
        nasal_murmur=None, voice_bar=False, breathy_offset=False),

    # ह  ha  — "haṃsadhvani" (rāg), "hai".  Glottal: no supralaryngeal
    # constriction, so locus is None -- the following vowel's own formants
    # are simply excited by the aspiration source.  This is Klatt's own
    # recipe for /h/ (it has no Table III entry): take the following vowel's
    # formants and bandwidths, widen B1 to about 300 Hz, and replace voicing
    # with aspiration.  The integration layer should apply that B1 widening;
    # there is no field for it here because it is a property of the vowel,
    # not of the consonant.  Hindi /h/ between vowels is typically
    # breathy-voiced rather than fully voiceless, hence breathy_offset=True
    # with voice_bar left False.
    'h': ConsonantSpec(
        ipa='h', place='glottal', manner='fricative', voiced=False, aspirated=True,
        locus=None, transition_dur=0.040,
        closure_dur=0.060, burst_dur=0.0, burst_formant=None,
        frication_db=-99.0, bypass_db=-99.0,   # aspiration source only
        aspiration_dur=0.090, aspiration_db=-6.0,
        nasal_murmur=None, voice_bar=False, breathy_offset=True),
}


# Conservative fallback for unknown IPA strings: an unaspirated voiceless
# dental stop with a mid, moderately diffuse burst.  Chosen because a short
# dental-ish stop is the least conspicuous thing to hear when the intended
# consonant is unknown -- it adds an onset without asserting a place cue as
# strongly as a labial or a retroflex would.
DEFAULT_SPEC = ConsonantSpec(
    ipa='?', place='dental', manner='stop', voiced=False, aspirated=False,
    locus=(250.0, 1700.0, 2600.0), transition_dur=0.050,
    closure_dur=0.065, burst_dur=0.008,
    burst_formant=(3000.0, 800.0),
    frication_db=-12.0, bypass_db=-16.0,
    aspiration_dur=0.015, aspiration_db=-18.0,
    nasal_murmur=None, voice_bar=False, breathy_offset=False)


# ---------------------------------------------------------------------------
# Derived specs: symbols outside the 33-item vocabulary that deserve their own
# parameters rather than a bare redirect.  Kept out of CONSONANTS so that dict
# stays exactly the server's inventory.
# ---------------------------------------------------------------------------

_DERIVED: Dict[str, ConsonantSpec] = {
    # ज़  za — voiced sibilant, in Perso-Arabic loans ("ġazal" ... "zamīn").
    # Built from /s/ rather than from /ɟ/: the place and noise spectrum are
    # those of /s/, with voicing added and the frication level dropped, since
    # voiced fricatives have weaker noise than their voiceless counterparts.
    'z': replace(CONSONANTS['s'], ipa='z', voiced=True,
                 frication_db=-12.0, bypass_db=-16.0, voice_bar=True),

    # ड़  ṛa — retroflex flap.  Same place cues as /ɖ/ (lowered F3) but a
    # ballistic, tap-length occlusion, like /r/'s.
    'ɽ': replace(CONSONANTS['ɖ'], ipa='ɽ',
                 closure_dur=0.022, burst_dur=0.005, transition_dur=0.035),

    # ढ़  ṛha — breathy retroflex flap ("paṛhnā").
    'ɽʱ': replace(CONSONANTS['ɖʱ'], ipa='ɽʱ',
                  closure_dur=0.025, burst_dur=0.005, transition_dur=0.035),

    # ण  ṇa — true retroflex nasal ("guṇ", "prāṇ", "vīṇā").  The server's
    # varga-15 slot is labelled न/"na", so CONSONANTS['n'] renders dentally;
    # the retroflex nasal lives here, with the lowered F3 that cues retroflexion
    # and Fujimura's retroflex antiformant range (2000-3000 Hz).
    'ɳ': replace(CONSONANTS['n'], ipa='ɳ', place='retroflex',
                 locus=(250.0, 1600.0, 2050.0),
                 nasal_murmur=(280.0, 90.0, 2200.0, 300.0)),
}


# ---------------------------------------------------------------------------
# Aliases: alternative IPA spellings and near-neighbour phonemes.  Each maps
# to a key in CONSONANTS or in _DERIVED.
# ---------------------------------------------------------------------------

ALIASES: Dict[str, str] = {
    # -- affricate spellings: Hindi च/ज are phonetically [tʃ]/[dʒ], and many
    #    transcribers write them that way (or with the tie-bar ligatures).
    'tʃ': 'c', 't͡ʃ': 'c', 'ʧ': 'c', 'tʃʰ': 'cʰ', 'ʧʰ': 'cʰ',
    'dʒ': 'ɟ', 'd͡ʒ': 'ɟ', 'ʤ': 'ɟ', 'dʒʱ': 'ɟʱ', 'ʤʱ': 'ɟʱ',
    # ʒ has no Hindi phoneme of its own; the nearest native sound is ज [dʒ].
    'ʒ': 'ɟ',

    # -- Perso-Arabic loan consonants, mapped to their nativised Hindi
    #    realisations (the standard substitutions for speakers without the
    #    loan phonology):
    'f': 'pʰ',    # फ़ -> फ; [f] and [pʰ] are in free variation for many speakers
    'q': 'k',     # क़ -> क (uvular -> velar)
    'x': 'kʰ',    # ख़ -> ख (velar fricative -> aspirated stop)
    'ɣ': 'g',     # ग़ -> ग
    'z': 'z',     # ज़ keeps its own derived spec (voiced /s/), see _DERIVED

    # -- flaps
    'ɽ': 'ɽ', 'ɽʱ': 'ɽʱ',    # own derived specs
    'ɾ': 'r',                # tap spelling of र
    'ɹ': 'r', 'ɻ': 'r',      # approximant r spellings

    # -- alternative nasal spellings.  The vocabulary's 'n' is labelled न and
    # renders dentally; ṇ routes to the retroflex spec in _DERIVED.
    'ṇ': 'ɳ',                # retroflex nasal
    'n̥': 'n̪', 'ɲ̥': 'ɲ',
    'ŋ̊': 'ŋ',
    'm̥': 'm',

    # -- retroflex/dental spelling variants
    'ʈ̪': 'ʈ', 't̪': 't', 't̪ʰ': 'tʰ', 'd̪': 'd', 'd̪ʱ': 'dʱ',
    'ʂ̪': 'ʂ', 'ʃ̪': 'ʃ',

    # -- approximant spellings
    'ʋ': 'v', 'w': 'v', 'β': 'v',
    'y': 'j',            # ASCII/romanisation habit: "ya" written with y
    'ɭ': 'l', 'ɫ': 'l',

    # -- breathy /h/ variants
    'ɦ': 'h', 'ʱ': 'h',

    # -- wrong-modifier repairs.  U+02B0 ʰ (voiceless aspiration) and U+02B1 ʱ
    #    (breathy voice) are routinely swapped when typing; redirect each
    #    mismatched pair to the phonologically correct member.
    'gʰ': 'gʱ', 'ɟʰ': 'ɟʱ', 'ɖʰ': 'ɖʱ', 'dʰ': 'dʱ', 'bʰ': 'bʱ',
    'kʱ': 'kʰ', 'cʱ': 'cʰ', 'ʈʱ': 'ʈʰ', 'tʱ': 'tʰ', 'pʱ': 'pʰ',

    # -- ASCII digraph romanisations (ITRANS-ish), for hand-typed data.
    'kh': 'kʰ', 'gh': 'gʱ', 'ch': 'cʰ', 'jh': 'ɟʱ',
    'th': 'tʰ', 'dh': 'dʱ', 'ph': 'pʰ', 'bh': 'bʱ',
    'j': 'j',       # exact key already; listed for clarity, never reached
    'sh': 'ʃ', 'ssh': 'ʂ', 'ng': 'ŋ', 'ny': 'ɲ',

    # -- ISO-15919 romanisation letters that do not collide with IPA keys.
    'ṭ': 'ʈ', 'ṭh': 'ʈʰ', 'ḍ': 'ɖ', 'ḍh': 'ɖʱ',
    'ś': 'ʃ', 'ṣ': 'ʂ', 'ñ': 'ɲ', 'ṅ': 'ŋ', 'ṛ': 'ɽ', 'ṛh': 'ɽʱ',
    'c': 'c',       # exact key already
}


def _strip_combining(text: str) -> str:
    """Drop combining diacritics (e.g. the U+032A dental bridge in 'n̪')."""
    return ''.join(ch for ch in unicodedata.normalize('NFD', text)
                   if not unicodedata.combining(ch))


def get_consonant(ipa: str) -> ConsonantSpec:
    """Look up a ConsonantSpec for an IPA string, tolerating spelling drift.

    Resolution order:
      1. exact key in CONSONANTS (the server's 33-symbol vocabulary),
      2. exact key in _DERIVED (z, ɽ, ɽʱ),
      3. ALIASES (alternative IPA spellings, loan phonemes, ASCII digraphs,
         swapped aspiration modifiers),
      4. the same steps again after Unicode NFC normalisation / lowercasing,
      5. the same steps again with combining diacritics stripped, so e.g.
         a mis-encoded 'n̪' still resolves,
      6. DEFAULT_SPEC.

    Never raises; unknown input yields DEFAULT_SPEC.
    """
    if not ipa:
        return DEFAULT_SPEC

    candidates = []
    raw = ipa.strip()
    candidates.append(raw)
    nfc = unicodedata.normalize('NFC', raw)
    candidates.append(nfc)
    candidates.append(nfc.lower())
    stripped = _strip_combining(nfc)
    candidates.append(stripped)
    candidates.append(stripped.lower())

    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        spec = CONSONANTS.get(cand) or _DERIVED.get(cand)
        if spec is not None:
            return spec
        target = ALIASES.get(cand)
        if target is not None:
            spec = CONSONANTS.get(target) or _DERIVED.get(target)
            if spec is not None:
                return spec
    return DEFAULT_SPEC
