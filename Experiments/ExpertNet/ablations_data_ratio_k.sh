echo "ExpertNet" >> Results/results_Baseline.txt
echo "FashionMNIST" >> Results/results_Baseline.txt

python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 2 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.001 >> Results/results_image_ablations.txt
python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 3 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.001 >> Results/results_image_ablations.txt
python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 5 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.001 >> Results/results_image_ablations.txt
python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 8 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.001 >> Results/results_image_ablations.txt
python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 10 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.001 >> Results/results_image_ablations.txt
python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 20 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.001 >> Results/results_image_ablations.txt

# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 2 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.01 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 3 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.01 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 5 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.01 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 8 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.01 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 10 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.01 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 20 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.01 >> Results/results_image_ablations.txt

# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 2 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.1 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 3 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.1 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 5 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.1 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 8 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.1 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 10 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.1 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 20 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.1 >> Results/results_image_ablations.txt

# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 2 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.5 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 3 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.5 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 5 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.5 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 8 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.5 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 10 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.5 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 20 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 0.5 >> Results/results_image_ablations.txt

# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 2 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 1 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 3 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 1 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 5 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 1 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 8 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 1 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 10 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 1 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 20 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 1 >> Results/results_image_ablations.txt

# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 2 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 2 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 3 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 2 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 5 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 2 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 8 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 2 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 10 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 2 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 20 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 2 >> Results/results_image_ablations.txt

# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 2 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 5 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 3 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 5 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 5 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 5 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 8 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 5 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 10 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 5 >> Results/results_image_ablations.txt
# python3 ../../expertnet.py --dataset FashionMNIST --n_runs 1 --n_clusters 20 --alpha 1 --beta 2 --gamma 1.5 --delta 0 --log_interval 2 --verbose False --n_epochs 100 --n_z 32 --n_classes 10 --verbose False --ae_type cnn --data_ratio 5 >> Results/results_image_ablations.txt