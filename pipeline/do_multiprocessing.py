from multiprocessing import Pool
import transit_search
import pandas as pd
import numpy as np

df_Astars = pd.read_csv("FINAL_SAMPLE.csv")
IDs = df_Astars.ID.values
Tmags = df_Astars.Tmag.values
Teffs = df_Astars.Teff.values
masses = df_Astars.mass.values
rads = df_Astars.rad.values
loggs = df_Astars.logg.values

func = transit_search.run_transit_search

def do_multiprocessing():
    i = np.arange(len(df_Astars))
    # set PROCESSES equal to the number of cores you want to use
    PROCESSES = 6
    with Pool(PROCESSES) as pool:
        results = pool.imap_unordered(func, i)
        print(results)
        for r in results:
            f = open("transit_search_results.csv", "a")
            if r[0] != 0:
                f.write(f"{r[0]}, {r[1]}, {r[2]}, {r[3]}, {r[4]}, {r[5]}, {r[6]}, {r[7]}, {r[8]}, {r[9]}, {r[10]}, {r[11]}\n")
            f.close()
        pool.close()

if __name__ == '__main__':
    do_multiprocessing()