echo "Classical Baseline" >> Results/results_Classical_Baseline.txt
echo "CIC" >> Results/results_Classical_Baseline.txt
python3 ../classical_baselines.py --dataset cic >> Results/results_Classical_Baseline.txt

echo "Sepsis" >> Results/results_Classical_Baseline.txt
python3 ../classical_baselines.py --dataset sepsis >> Results/results_Classical_Baseline.txt

echo "Kidney" >> Results/results_Classical_Baseline.txt
python3 ../classical_baselines.py --dataset aki >> Results/results_Classical_Baseline.txt

echo "Respiratory" >> Results/results_Classical_Baseline.txt
python3 ../classical_baselines.py --dataset ards >> Results/results_Classical_Baseline.txt

echo "Wid_Mortality" >> Results/results_Classical_Baseline.txt
python3 ../classical_baselines.py --dataset wid_mortality >> Results/results_Classical_Baseline.txt

echo "Diabetes" >> Results/results_Classical_Baseline.txt
python3 ../classical_baselines.py --dataset diabetes >> Results/results_Classical_Baseline.txt

echo "CIC-LOS" >> Results/results_Classical_Baseline.txt
python3 ../classical_baselines.py --dataset cic_los >> Results/results_Classical_Baseline.txt

echo "Kidney New" >> Results/results_Classical_Baseline.txt
python3 ../classical_baselines.py --dataset aki_new >> Results/results_Classical_Baseline.txt

echo "Respiratory New" >> Results/results_Classical_Baseline.txt
python3 ../classical_baselines.py --dataset ards_new >> Results/results_Classical_Baseline.txt

echo "CIC New" >> Results/results_Classical_Baseline.txt
python3 ../classical_baselines.py --dataset cic_new >> Results/results_Classical_Baseline.txt