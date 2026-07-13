"""Demo: generate pitch prevalence visualizations in all modes from live IDTAP data."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from idtap import SwaraClient, Piece
from idtap.visualization import plot_pitch_prevalence, plot_pitch_patterns

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'demo_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Piece IDs for two good transcriptions
PIECE_IDS = {
    'mak_yaman': '63445d13dc8b9023a09747a6',   # Mushtaq Ali Khan - Yaman
    'vk_yaman':  '63595e5518b58cbd4ce4740e',   # Vilayat Khan - Yaman (LP) DN
}


def main():
    print("Connecting to IDTAP...")
    client = SwaraClient()
    print("Authenticated.\n")

    # Use Mushtaq Ali Khan Yaman (same as screenshot) for primary demos
    piece_id = PIECE_IDS['mak_yaman']
    print(f"Fetching Mushtaq Ali Khan - Yaman ({piece_id})...")
    piece_data = client.get_piece(piece_id)
    # Strip any keys the Piece class doesn't recognize yet
    known_keys = {
        '_id', 'adHocSectionCatGrid', 'assemblageDescriptors', 'audioID',
        'audio_DB_ID', 'collections', 'dateCreated', 'dateModified',
        'durArray', 'durArrayGrid', 'durTot', 'excerptRange',
        'explicitPermissions', 'family_name', 'given_name',
        'instrumentation', 'location', 'meters', 'name', 'permissions',
        'phraseGrid', 'phrases', 'raga', 'sectionCatGrid',
        'sectionCategorization', 'sectionStarts', 'sectionStartsGrid',
        'soloInstrument', 'soloist', 'title', 'trackTitles', 'userID',
        'publicView',
    }
    # Also drop collections if it has non-string items (server schema mismatch)
    if 'collections' in piece_data:
        if not all(isinstance(c, str) for c in piece_data.get('collections', [])):
            del piece_data['collections']
    piece_data = {k: v for k, v in piece_data.items() if k in known_keys}
    piece = Piece.from_json(piece_data)
    piece_title = piece.title or 'Mushtaq Ali Khan - Yaman'
    trajs = piece.all_trajectories(inst=0)
    print(f"  Trajectories: {len(trajs)}, Duration: {piece.dur_tot:.0f}s\n")

    configs = [
        # --- Section segmentation ---
        {
            'label': '01_section_standard',
            'desc': 'Section / Standard / pitchNumber',
            'kwargs': dict(
                segmentation='section', output_type='pitchNumber',
                pitch_representation='fixed_pitch',
                condensed=False, heatmap=False,
            ),
        },
        {
            'label': '02_section_condensed',
            'desc': 'Section / Standard / pitchNumber / Condensed',
            'kwargs': dict(
                segmentation='section', output_type='pitchNumber',
                pitch_representation='fixed_pitch',
                condensed=True, heatmap=False,
            ),
        },
        {
            'label': '03_section_heatmap',
            'desc': 'Section / Heatmap / pitchNumber',
            'kwargs': dict(
                segmentation='section', output_type='pitchNumber',
                pitch_representation='fixed_pitch',
                condensed=False, heatmap=True,
            ),
        },
        {
            'label': '04_section_heatmap_condensed',
            'desc': 'Section / Heatmap / pitchNumber / Condensed',
            'kwargs': dict(
                segmentation='section', output_type='pitchNumber',
                pitch_representation='fixed_pitch',
                condensed=True, heatmap=True,
            ),
        },
        {
            'label': '05_section_chroma',
            'desc': 'Section / PitchChroma / chroma',
            'kwargs': dict(
                segmentation='section', output_type='chroma',
                pitch_representation='fixed_pitch',
                condensed=False, heatmap=False,
            ),
        },
        {
            'label': '06_section_chroma_heatmap',
            'desc': 'Section / PitchChroma / chroma / Heatmap',
            'kwargs': dict(
                segmentation='section', output_type='chroma',
                pitch_representation='fixed_pitch',
                condensed=False, heatmap=True,
            ),
        },
        {
            'label': '07_section_pitch_onsets',
            'desc': 'Section / Pitch Onsets / pitchNumber / Condensed',
            'kwargs': dict(
                segmentation='section', output_type='pitchNumber',
                pitch_representation='pitch_onsets',
                condensed=True, heatmap=False,
            ),
        },
        # --- Phrase segmentation ---
        {
            'label': '08_phrase_standard',
            'desc': 'Phrase / Standard / pitchNumber / Condensed',
            'kwargs': dict(
                segmentation='phrase', output_type='pitchNumber',
                pitch_representation='fixed_pitch',
                condensed=True, heatmap=False,
            ),
        },
        {
            'label': '09_phrase_heatmap',
            'desc': 'Phrase / Heatmap / pitchNumber / Condensed',
            'kwargs': dict(
                segmentation='phrase', output_type='pitchNumber',
                pitch_representation='fixed_pitch',
                condensed=True, heatmap=True,
            ),
        },
        # --- Duration segmentation ---
        {
            'label': '10_duration_10s',
            'desc': 'Duration (10s) / Standard / pitchNumber / Condensed',
            'kwargs': dict(
                segmentation='duration', output_type='pitchNumber',
                pitch_representation='fixed_pitch',
                segment_duration=10,
                condensed=True, heatmap=False,
            ),
        },
        {
            'label': '11_duration_10s_heatmap',
            'desc': 'Duration (10s) / Heatmap / pitchNumber / Condensed',
            'kwargs': dict(
                segmentation='duration', output_type='pitchNumber',
                pitch_representation='fixed_pitch',
                segment_duration=10,
                condensed=True, heatmap=True,
            ),
        },
        {
            'label': '12_duration_30s_chroma',
            'desc': 'Duration (30s) / Chroma / Heatmap',
            'kwargs': dict(
                segmentation='duration', output_type='chroma',
                pitch_representation='fixed_pitch',
                segment_duration=30,
                condensed=False, heatmap=True,
            ),
        },
    ]

    print(f"Generating {len(configs)} visualizations...\n")

    for cfg in configs:
        label = cfg['label']
        desc = cfg['desc']
        kwargs = cfg['kwargs']

        print(f"  [{label}] {desc}...", end=' ', flush=True)
        try:
            fig = plot_pitch_prevalence(piece, title=piece_title, **kwargs)
            path = os.path.join(OUTPUT_DIR, f'{label}.png')
            fig.savefig(path, dpi=300, bbox_inches='tight',
                        pad_inches=0.2, facecolor='white', edgecolor='none')
            plt.close(fig)
            print(f"OK → {path}")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Pitch pattern visualizations
    # ------------------------------------------------------------------
    pattern_configs = [
        {
            'label': '13_patterns_3gram',
            'desc': 'Patterns / 3-gram / pitchNumber',
            'kwargs': dict(pattern_size=3, output_type='pitchNumber'),
        },
        {
            'label': '14_patterns_3gram_plot',
            'desc': 'Patterns / 3-gram / pitchNumber / contour',
            'kwargs': dict(pattern_size=3, output_type='pitchNumber', plot=True),
        },
        {
            'label': '15_patterns_multi_section',
            'desc': 'Patterns / [2,3,4]-gram / section segmentation',
            'kwargs': dict(
                pattern_sizes=[2, 3, 4], segmentation='section',
                output_type='pitchNumber', max_patterns=10,
            ),
        },
        {
            'label': '16_patterns_sargam',
            'desc': 'Patterns / 3-gram / sargamLetter',
            'kwargs': dict(pattern_size=3, output_type='sargamLetter'),
        },
    ]

    print(f"\nGenerating {len(pattern_configs)} pattern visualizations...\n")

    for cfg in pattern_configs:
        label = cfg['label']
        desc = cfg['desc']
        kwargs = cfg['kwargs']

        print(f"  [{label}] {desc}...", end=' ', flush=True)
        try:
            fig = plot_pitch_patterns(piece, title=piece_title, **kwargs)
            path = os.path.join(OUTPUT_DIR, f'{label}.png')
            fig.savefig(path, dpi=300, bbox_inches='tight',
                        pad_inches=0.2, facecolor='white', edgecolor='none')
            plt.close(fig)
            print(f"OK → {path}")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
