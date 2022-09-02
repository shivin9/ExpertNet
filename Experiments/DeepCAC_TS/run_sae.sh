# echo "SAE" >> Results/results_SAE.txt
# echo "CIC" >> Results/results_SAE.txt
# python3 ../../baseline.py --dataset cic --n_epochs 100 --alpha 1 --gamma 0.01 --n_runs 5 >> Results/results_SAE.txt

# echo "Sepsis" >> Results/results_SAE.txt
# python3 ../../baseline.py --dataset sepsis --n_epochs 100 --alpha 1 --gamma 0.01 --n_runs 5 >> Results/results_SAE.txt

# echo "Kidney" >> Results/results_SAE.txt
# python3 ../../baseline.py --dataset aki --n_epochs 100 --alpha 1 --gamma 0.01 --n_runs 5 >> Results/results_SAE.txt

# echo "Respiratory" >> Results/results_SAE.txt
# python3 ../../baseline.py --dataset ards --n_epochs 100 --alpha 1 --gamma 0.01 --n_runs 5 >> Results/results_SAE.txt

# echo "Wid_Mortality" >> Results/results_SAE.txt
# python3 ../../baseline.py --dataset wid_mortality --n_epochs 100 --alpha 1 --gamma 0.01 --n_runs 5 >> Results/results_SAE.txt

# echo "Diabetes" >> Results/results_SAE.txt
# python3 ../../baseline.py --dataset diabetes --n_epochs 100 --alpha 1 --gamma 0.01 --n_runs 5 >> Results/results_SAE.txt

echo "CIC-LOS" >> Results/results_SAE.txt
python3 ../../baseline.py --dataset cic_los --n_epochs 100 --alpha 1 --gamma 0.01 --n_runs 5 --n_classes 3 >> Results/results_SAE.txt

echo "Kidney New" >> Results/results_SAE.txt
python3 ../../baseline.py --dataset aki_new --n_epochs 100 --alpha 1 --gamma 0.01 --n_runs 5 >> Results/results_SAE.txt

echo "Respiratory New" >> Results/results_SAE.txt
python3 ../../baseline.py --dataset ards_new --n_epochs 100 --alpha 1 --gamma 0.01 --n_runs 5 >> Results/results_SAE.txt

echo "CIC New" >> Results/results_SAE.txt
python3 ../../baseline.py --dataset cic_new --n_epochs 100 --alpha 1 --gamma 0.01 --n_runs 5 >> Results/results_SAE.txt