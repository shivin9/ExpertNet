echo "Baseline" >> Results/results_Baseline.txt
echo "CIC" >> Results/results_Baseline.txt
python3 baseline.py --dataset cic --n_epochs 50 >> Results/results_Baseline.txt

echo "Sepsis" >> Results/results_Baseline.txt
python3 baseline.py --dataset sepsis --n_epochs 50 >> Results/results_Baseline.txt

echo "Kidney" >> Results/results_Baseline.txt
python3 baseline.py --dataset aki --n_epochs 50 >> Results/results_Baseline.txt

echo "Respiratory" >> Results/results_Baseline.txt
python3 baseline.py --dataset ards --n_epochs 50 >> Results/results_Baseline.txt

echo "Wid_Mortality" >> Results/results_Baseline.txt
python3 baseline.py --dataset wid_mortality --n_epochs 50 >> Results/results_Baseline.txt

echo "Diabetes" >> Results/results_Baseline.txt
python3 baseline.py --dataset diabetes --n_epochs 50 >> Results/results_Baseline.txt