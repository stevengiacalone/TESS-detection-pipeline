from multiprocessing import Pool
import injection_recovery
import pandas as pd
import numpy as np
import lightkurve as lk

df_Astars = pd.read_csv("RANDOM_SAMPLE.csv")

func = injection_recovery.run_injection_recovery

def do_multiprocessing():

    for i in range(len(df_Astars)):
        res = injection_recovery.get_lightcurve(i)
        if res is None:
            continue
        else:
            this_lc, max_exptime, this_ID, this_Tmag, this_Teff, this_mass, this_rad, this_logg = res
            print(i)

            j = np.arange(50)
            # update these to inject planets with different properties
            this_per = np.random.uniform(low=0.5, high=10.0, size=50)
            this_t0 = this_lc.time.value.min() + this_per*np.random.random(size=50)
            this_rp = np.random.uniform(low=1.0, high=8.0, size=50)
            this_b = np.random.uniform(low=0.0, high=0.9, size=50)

            # set PROCSSES to number of cores to use
            PROCESSES = 6
            with Pool(PROCESSES) as pool:
                results = pool.imap_unordered(
                    func, 
                    [(j[idx], this_lc, max_exptime, int(this_ID), this_Tmag, this_Teff, this_mass, this_rad, this_logg, this_per[idx], this_t0[idx], this_rp[idx], this_b[idx]) for idx in range(50)]
                    )
                for r in results:
                    f = open("injection_recovery_results.csv", "a")
                    f.write(f"{r[0]}, {r[1]}, {r[2]}, {r[3]}, {r[4]}, {r[5]}, {r[6]}, {r[7]}, {r[8]}, {r[9]}, {r[10]}, {r[11]}, {r[12]}, {r[13]}, {r[14]}, {r[15]}, {r[16]}, {r[17]}, {r[18]}, {r[19]}\n")
                    f.close()

if __name__ == '__main__':
    do_multiprocessing()
