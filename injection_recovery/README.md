To run the injection-recovery test, follow these steps:
1. Create a file named RANDOM_SAMPLE.csv in the same format as the one in this folder. This csv should contain all of the stars you want to include in the tests.
2. Update the injection_recovery.py file to fit the needs of your science case. For example, it currently rejects any signals corresponding to a planet radius greater than 8 Earth radii. If you want to find larger planets, you'll need to update this number. You may also want to tune the parameters of the BLS to speed things up.
3. Update PROCESSES in do_multiprocessing.py to the number of cores you want to use. Also update the properties of the injected planets, if desired.
4. Run the parallelized search by entering "python do_multiprocessing.py" in the command line.

The included file RANDOM_SAMPLE.csv corresponds to the random stellar sample used for injection-recovery tests in Giacalone & Dressing (2024).
