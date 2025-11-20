def align(seq1, seq2, match=1, gap_penalty=1, mismatch_penalty=1):
    m = len(seq1)
    n = len(seq2)

    # Initialize and calculate scoring matrix
    score = [[0 for j in range(n+1)] for i in range(m+1)]
    max_score = 0
    max_pos = (0, 0)
    for i in range(1, m+1):
        for j in range(1, n+1):
            if seq1[i-1] == seq2[j-1]:
                diag = score[i-1][j-1] + match
            else:
                diag = score[i-1][j-1] - mismatch_penalty
            up = score[i-1][j] - gap_penalty
            left = score[i][j-1] - gap_penalty
            score[i][j] = max(0, diag, up, left)
            if score[i][j] > max_score:
                max_score = score[i][j]
                max_pos = (i, j)

    # Traceback
    aligned_seq1 = []
    aligned_seq2 = []
    i, j = max_pos
    while score[i][j] != 0: 
        if score[i][j] == score[i-1][j] - gap_penalty:
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append('-')
            i -= 1
        elif score[i][j] == score[i][j-1] - gap_penalty:
            aligned_seq1.append('-')
            aligned_seq2.append(seq2[j-1])
            j -= 1
        else:
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append(seq2[j-1])
            i -= 1
            j -= 1

    aligned_seq1.reverse()
    aligned_seq2.reverse()

    # for row in score:
    #     print(row)  

    return ''.join(aligned_seq1), ''.join(aligned_seq2), max_score

print(align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac'))

print(align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', gap_penalty=2))

print(align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', mismatch_penalty=0.3))