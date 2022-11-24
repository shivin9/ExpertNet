# echo "Baseline" >> Results/results_Baseline.txt
# echo "CIC" >> Results/results_Baseline.txt
# python3 ../../baseline.py --dataset cic --n_epochs 100 --gamma 1 --n_runs 5 >> Results/results_Baseline.txt

# echo "Kidney" >> Results/results_Baseline.txt
# python3 ../../baseline.py --dataset aki --n_epochs 100 --gamma 1 --n_runs 5 >> Results/results_Baseline.txt

# echo "Respiratory" >> Results/results_Baseline.txt
# python3 ../../baseline.py --dataset ards --n_epochs 100 --gamma 1 --n_runs 5 >> Results/results_Baseline.txt

# echo "Wid_Mortality" >> Results/results_Baseline.txt
# python3 ../../baseline.py --dataset wid_mortality --n_epochs 100 --gamma 1 --n_runs 5 >> Results/results_Baseline.txt

# echo "Diabetes" >> Results/results_Baseline.txt
# python3 ../../baseline.py --dataset diabetes --n_epochs 100 --gamma 1 --n_runs 5 >> Results/results_Baseline.txt

# echo "CIC-LOS" >> Results/results_Baseline.txt
# python3 ../../baseline.py --dataset cic_los_new --n_epochs 100 --gamma 1 --n_runs 5 --n_classes 3 >> Results/results_Baseline.txt

# echo "Kidney New" >> Results/results_Baseline.txt
# python3 ../../baseline.py --dataset aki_new --n_epochs 100 --gamma 1 --n_runs 5 >> Results/results_Baseline.txt

# echo "Respiratory New" >> Results/results_Baseline.txt
# python3 ../../baseline.py --dataset ards_new --n_epochs 100 --gamma 1 --n_runs 5 >> Results/results_Baseline.txt

# echo "CIC New" >> Results/results_Baseline.txt
# python3 ../../baseline.py --dataset cic_new --n_epochs 100 --gamma 1 --n_runs 5 >> Results/results_Baseline.txt

python3 ../../baseline.py --dataset FashionMNIST --n_runs 1 --verbose False --n_epochs 100 --ae_type cnn --n_classes 10 --data_ratio 0.01 >> Results/results_Baseline.txt
python3 ../../baseline.py --dataset FashionMNIST --n_runs 1 --verbose False --n_epochs 100 --ae_type cnn --n_classes 10 --data_ratio 0.1 >> Results/results_Baseline.txt
python3 ../../baseline.py --dataset FashionMNIST --n_runs 1 --verbose False --n_epochs 100 --ae_type cnn --n_classes 10 --data_ratio 0.25 >> Results/results_Baseline.txt
python3 ../../baseline.py --dataset FashionMNIST --n_runs 1 --verbose False --n_epochs 100 --ae_type cnn --n_classes 10 --data_ratio 0.40 >> Results/results_Baseline.txt
python3 ../../baseline.py --dataset FashionMNIST --n_runs 1 --verbose False --n_epochs 100 --ae_type cnn --n_classes 10 --data_ratio 0.50 >> Results/results_Baseline.txt
python3 ../../baseline.py --dataset FashionMNIST --n_runs 1 --verbose False --n_epochs 100 --ae_type cnn --n_classes 10 --data_ratio 0.75 >> Results/results_Baseline.txt
python3 ../../baseline.py --dataset FashionMNIST --n_runs 1 --verbose False --n_epochs 100 --ae_type cnn --n_classes 10 --data_ratio 1.0 >> Results/results_Baseline.txt