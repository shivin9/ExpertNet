# echo "Classical Baseline" >> Results/results_Classical_Baseline.txt

# echo "Titanic" >> Results/results_Classical_Baseline.txt
# python3 ../../classical_baselines.py --dataset titanic >> Results/results_Classical_Baseline.txt

# echo "Heart" >> Results/results_Classical_Baseline.txt
# python3 ../../classical_baselines.py --dataset heart >> Results/results_Classical_Baseline.txt

# echo "Creditcard" >> Results/results_Classical_Baseline.txt
# python3 ../../classical_baselines.py --dataset creditcard >> Results/results_Classical_Baseline.txt

# echo "Diabetes" >> Results/results_Classical_Baseline.txt
# python3 ../../classical_baselines.py --dataset diabetes >> Results/results_Classical_Baseline.txt

# echo "Adult" >> Results/results_Classical_Baseline.txt
# python3 ../../classical_baselines.py --dataset adult >> Results/results_Classical_Baseline.txt

# echo "CIC" >> Results/results_Classical_Baseline.txt
# python3 ../../classical_baselines.py --dataset cic >> Results/results_Classical_Baseline.txt

# echo "CIC_LOS" >> Results/results_Classical_Baseline.txt
# python3 ../../classical_baselines.py --dataset cic_los --n_classes 3 --verbose True >> Results/results_Classical_Baseline.txt

# echo "Sepsis" >> Results/results_Classical_Baseline.txt
# python3 ../../classical_baselines.py --dataset sepsis >> Results/results_Classical_Baseline.txt

# echo "Kidney" >> Results/results_Classical_Baseline.txt
# python3 ../../classical_baselines.py --dataset aki >> Results/results_Classical_Baseline.txt

# echo "Respiratory" >> Results/results_Classical_Baseline.txt
# python3 ../../classical_baselines.py --dataset ards >> Results/results_Classical_Baseline.txt

# echo "Wid_Mortality" >> Results/results_Classical_Baseline.txt
# python3 ../../classical_baselines.py --dataset wid_mortality >> Results/results_Classical_Baseline.txt

echo "Kidney New" >> Results/results_Classical_Baseline.txt
python3 ../../classical_baselines.py --dataset aki_new >> Results/results_Classical_Baseline.txt

echo "Respiratory New" >> Results/results_Classical_Baseline.txt
python3 ../../classical_baselines.py --dataset ards_new >> Results/results_Classical_Baseline.txt

echo "CIC New" >> Results/results_Classical_Baseline.txt
python3 ../../classical_baselines.py --dataset cic_new >> Results/results_Classical_Baseline.txt

echo "CIC LOS New" >> Results/results_Classical_Baseline.txt
python3 ../../classical_baselines.py --dataset cic_los_new --n_classes 3 --verbose True >> Results/results_Classical_Baseline.txt

echo "IHM" >> Results/results_Classical_Baseline.txt
python3 ../../classical_baselines.py --dataset ihm >> Results/results_Classical_Baseline.txt
