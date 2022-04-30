# echo "Baseline" >> Results/results_Baseline.txt
# echo "CIC" >> Results/results_Baseline.txt
# python3 ../baseline.py --dataset cic --n_epochs 50 --gamma 1 --n_runs 11 >> Results/results_Baseline.txt

# echo "Sepsis" >> Results/results_Baseline.txt
# python3 ../baseline.py --dataset sepsis --n_epochs 50 --gamma 1 --n_runs 11 >> Results/results_Baseline.txt

# echo "Kidney" >> Results/results_Baseline.txt
# python3 ../baseline.py --dataset aki --n_epochs 50 --gamma 1 --n_runs 11 >> Results/results_Baseline.txt

# echo "Respiratory" >> Results/results_Baseline.txt
# python3 ../baseline.py --dataset ards --n_epochs 50 --gamma 1 --n_runs 11 >> Results/results_Baseline.txt

# echo "Wid_Mortality" >> Results/results_Baseline.txt
# python3 ../baseline.py --dataset wid_mortality --n_epochs 50 --gamma 1 --n_runs 11 >> Results/results_Baseline.txt

# echo "Diabetes" >> Results/results_Baseline.txt
# python3 ../baseline.py --dataset diabetes --n_epochs 50 --gamma 1 --n_runs 11 >> Results/results_Baseline.txt

# echo "CIC-LOS" >> Results/results_Baseline.txt
# python3 ../baseline.py --dataset cic_los --n_epochs 50 --gamma 1 --n_runs 11 --n_classes 3 >> Results/results_Baseline.txt

# echo "Kidney New" >> Results/results_Baseline.txt
# python3 ../baseline.py --dataset aki_new --n_epochs 50 --gamma 1 --n_runs 11 >> Results/results_Baseline.txt

# echo "Respiratory New" >> Results/results_Baseline.txt
# python3 ../baseline.py --dataset ards_new --n_epochs 50 --gamma 1 --n_runs 11 >> Results/results_Baseline.txt

# echo "CIC New" >> Results/results_Baseline.txt
# python3 ../baseline.py --dataset cic_new --n_epochs 50 --gamma 1 --n_runs 11 >> Results/results_Baseline.txt

########################################################################################
# DeepCAC Baselines
########################################################################################
echo "Baseline" >> Results/results_Baseline.txt
echo "CIC" >> Results/results_Baseline.txt
python3 ../baseline.py --dataset cic --n_epochs 50 --gamma 1 --n_runs 5 >> Results/results_Baseline.txt

echo "Diabetes" >> Results/results_Baseline.txt
python3 ../baseline.py --dataset diabetes --n_epochs 50 --gamma 1 --n_runs 5 >> Results/results_Baseline.txt

echo "Adult" >> Results/results_Baseline.txt
python3 ../baseline.py --dataset adult --n_epochs 50 --gamma 1 --n_runs 5 >> Results/results_Baseline.txt

echo "Titanic" >> Results/results_Baseline.txt
python3 ../baseline.py --dataset titanic --n_epochs 50 --gamma 1 --n_runs 5 >> Results/results_Baseline.txt

echo "Heart" >> Results/results_Baseline.txt
python3 ../baseline.py --dataset heart --n_epochs 50 --gamma 1 --n_runs 5 >> Results/results_Baseline.txt

echo "Creditcard" >> Results/results_Baseline.txt
python3 ../baseline.py --dataset creditcard --n_epochs 50 --gamma 1 --n_runs 5 >> Results/results_Baseline.txt

echo "CIC-LOS" >> Results/results_Baseline.txt
python3 ../baseline.py --dataset cic_los --n_epochs 50 --gamma 1 --n_runs 5 --n_classes 3 >> Results/results_Baseline.txt
