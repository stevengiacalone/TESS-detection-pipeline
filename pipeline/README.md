To run this planet search pipeline, follow these steps:
1. Create a file named FINAL_SAMPLE.csv in the same format as the one in this folder. This csv should contain all of the stars you want to search.
2. Update the transit_search.py file to fit the needs of your science case. For example, it currently rejects any signals corresponding to a planet radius greater than 8 Earth radii. If you want to find larger planets, you'll need to update this number. You may also want to tune the parameters of the BLS to speed things up.
3. Update PROCESSES in do_multiprocessing.py to the number of cores you want to use.
4. Run the parallelized search by entering "python do_multiprocessing.py" in the command line.

The included files FINAL_SAMPLE.csv and transit_search_results.py correspond to the stellar sample and list of threshold-crossing events discussed in Giacalone & Dressing (2024).
