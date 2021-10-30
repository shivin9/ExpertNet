echo "Baseline" >> results_Baseline.txt
echo "CIC" >> results_Baseline.txt
python3 baseline.py --dataset cic --n_epochs 50 >> results_Baseline.txt

echo "Sepsis" >> results_Baseline.txt
python3 baseline.py --dataset sepsis --n_epochs 50 >> results_Baseline.txt

echo "Kidney" >> results_Baseline.txt
python3 baseline.py --dataset aki --n_epochs 50 >> results_Baseline.txt

echo "Respiratory" >> results_Baseline.txt
python3 baseline.py --dataset ards --n_epochs 50 >> results_Baseline.txt

echo "Wid_Mortality" >> results_Baseline.txt
python3 baseline.py --dataset wid_mortality --n_epochs 50 >> results_Baseline.txt

echo "Diabetes" >> results_Baseline.txt
python3 baseline.py --dataset diabetes --n_epochs 50 >> results_Baseline.txt