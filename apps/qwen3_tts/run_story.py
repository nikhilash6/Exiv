#!/usr/bin/env python3
"""
Simple wrapper to run hook_app.py with the Moon-Tender story
"""

from hook_app import handle_generate

# The Moon-Tender of Aethyria story
STORY_TEXT = """The last Moon-Tender of Aethyria climbed the glass mountain at twilight, her ladder of braided starlight slung across one shoulder. Lyra had inherited the position when her grandmother's hands grew too spotted and trembling to polish the lunar surface—a duty passed down through generations that the modern world had forgotten, then dismissed as fairy tale.

Tonight was the Waning, the most dangerous phase.

She reached the summit where the moon sat lodged in a crater of its own making, no longer distant and celestial but close enough to touch, pale and pitted like ancient bone. It hummed, as it always did, a frequency that vibrated in her teeth and behind her eyes. The moon was not a rock, she had learned as a child, but a *repository*. Every dream deferred, every secret whispered at midnight, every spell cast in desperation—the moon absorbed them all, storing them in its hollows like rainwater in stone crevices.

But repositories can overflow.

Lyra unrolled her ladder and began to climb the curved face. The silver dust there coated her fingers immediately, cold and slightly greasy, the residue of ten thousand unspoken confessions. She reached the first fissure—a dark line bisecting the Sea of Tranquility—and dipped her brush into the jar at her hip. The ink within was distilled from her own breath and morning dew, the only solvent gentle enough for this work.

She painted along the crack, whispering the old words:
"""

if __name__ == "__main__":
    result = handle_generate(
        text=STORY_TEXT,
        ref_audio_id="calm_male",  # Change this to your preferred voice
        language="English",
        enable_chunking=True,
        chunk_size=20
    )
    print(f"\nResult: {result}")
