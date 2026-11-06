# Sandbox look: popcorn with no crystal, but the push rises the longer it waits (inner-horizon pileup on a still X).
import numpy as np
rng=np.random.default_rng(7)
for tau in [None, 200, 60, 20]:
    cvs=[]
    for s in range(200):
        gaps=[]
        for _ in range(100):
            t=0
            while True:
                t+=1
                h=(1/110)*(1 if tau is None else np.exp(t/tau))/(1 if tau is None else 3)
                if rng.random()<min(h,1): break
            gaps.append(t)
        g=np.array(gaps); cvs.append(g.std()/g.mean())
    print(f"still X, push rise time {str(tau):>5}: mean gap {np.mean(g):6.1f}  CV {np.mean(cvs):.2f}")
