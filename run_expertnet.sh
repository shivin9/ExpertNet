echo "ExpertNet" >> results_EN.txt
echo "CIC" >> results_EN.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_EN.txt
	python3 deepcac.py --dataset cic --n_clusters $j --alpha 1 --beta 0.5 --gamma 1.5 --delta 0.01 --log_interval 2 --n_epochs 50
done

echo "ExpertNet" >> results_EN.txt
echo "Sepsis" >> results_EN.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_EN.txt
	python3 deepcac.py --dataset sepsis --n_clusters $j --alpha 1 --beta 0.5 --gamma 1.5 --delta 0.01 --log_interval 2 --n_epochs 50
done

echo "ExpertNet" >> results_EN.txt
echo "Kidney" >> results_EN.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_EN.txt
	python3 deepcac.py --dataset kidney --n_clusters $j --alpha 1 --beta 0.5 --gamma 1.5 --delta 0.01 --log_interval 2 --n_epochs 50
done

echo "ExpertNet" >> results_EN.txt
echo "Respiratory" >> results_EN.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_EN.txt
	python3 deepcac.py --dataset respiratory --n_clusters $j --alpha 1 --beta 0.5 --gamma 1.5 --delta 0.01 --log_interval 2 --n_epochs 50
done

echo "ExpertNet" >> results_EN.txt
echo "Wid_Mortality" >> results_EN.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_EN.txt
	python3 deepcac.py --dataset wid_mortality --n_clusters $j --alpha 1 --beta 0.5 --gamma 1.5 --delta 0.01 --log_interval 2 --n_epochs 50
done

echo "ExpertNet" >> results_EN.txt
echo "Diabetes" >> results_EN.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_EN.txt
	python3 deepcac.py --dataset diabetes --n_clusters $j --alpha 1 --beta 0.5 --gamma 1.5 --delta 0.01 --log_interval 2 --n_epochs 50
done