import os

mpi_root = 'E:/Thesis/mpi_inf_3dhp'

for subj in ['S1', 'S2']:
    for seq in ['Seq1', 'Seq2']:
        imgseq = os.path.join(mpi_root, subj, seq, 'imageSequence')
        if not os.path.exists(imgseq):
            continue
        contents = sorted(os.listdir(imgseq))
        print(f'{subj}/{seq}/imageSequence:')
        for d in contents:
            dd = os.path.join(imgseq, d)
            if os.path.isdir(dd):
                frames = sorted([f for f in os.listdir(dd) if f.endswith('.jpg')])
                n = len(frames)
                first = frames[0] if n > 0 else "empty"
                last = frames[-1] if n > 0 else "empty"
                print(f'  {d}: {n} frames ({first} ... {last})')
