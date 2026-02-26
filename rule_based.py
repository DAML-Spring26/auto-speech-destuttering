def rule_detect(text):

    words = text.lower().split()

    # REP detection
    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            return 0
        
    # PAU proxyn for very short text
    if len(words) <= 1:
        return 3

    return -1  # if unknown