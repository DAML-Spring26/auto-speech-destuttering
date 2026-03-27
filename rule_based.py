import re
import random

def rule_detect(text):
    """
    Rule-based detection of speech disfluencies.

    Classes:
        0: REP (repetition)
        3: PAU (pause/filler)
        4: SUB (substitution / noisy word)
    """

    text_clean = text.lower().strip()
    words = [re.sub(r"[^\w]", "", w) for w in text_clean.split() if w]

    # Safety: empty input
    if len(words) == 0:
        return 3

    # 1️⃣ REP: repeated or near-repeated words
    for i in range(1, len(words)):
        # exact repetition
        if words[i] == words[i-1]:
            return 0

        # skip-word repetition: "I ... I"
        if i > 1 and words[i] == words[i-2]:
            return 0

        # stutter pattern: "th-the"
        if len(words[i]) > 2 and words[i].startswith(words[i-1][:2]):
            return 0

        # repeated multiple times in sentence
        if words.count(words[i]) > 2:
            return 0

    # 2️⃣ PAU: filler words
    fillers = {
        "uh", "uhh", "um", "umm", "erm", "ermm",
        "ah", "eh", "mm", "hmm",
        "like", "youknow"
    }

    for w in words:
        if w in fillers:
            return 3

    # PAU: pause patterns
    if "..." in text_clean:
        return 3

    # PAU: very short filler-only utterance
    if len(words) == 1 and words[0] in fillers:
        return 3

    # 3️⃣ SUB: abnormal word patterns
    for w in words:
        # repeated vowels (very strict)
        if re.search(r"(a|e|i|o|u)\1{4,}", w):
            return 4

        # extremely long word (likely ASR error)
        if len(w) > 18:
            return 4

        # strong non-alphabetic noise
        if re.search(r"[^a-z]{3,}", w):
            return 4

    # 4️⃣ fallback: weighted to reduce bias but keep accuracy
    return random.choices([4, 3], weights=[0.8, 0.2])[0]