echo "KMeans" >> results_KM.txt
echo "CIC" >> results_KM.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_KM.txt
	python3 KMeans.py --dataset cic --n_clusters $j --log_interval 2 --n_epochs 50 >> results_KM.txt
done

echo "KMeans" >> results_KM.txt
echo "Sepsis" >> results_KM.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_KM.txt
	python3 KMeans.py --dataset sepsis --n_clusters $j --log_interval 2 --n_epochs 50 >> results_KM.txt
done

echo "KMeans" >> results_KM.txt
echo "Kidney" >> results_KM.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_KM.txt
	python3 KMeans.py --dataset kidney --n_clusters $j --log_interval 2 --n_epochs 50 >> results_KM.txt
done

echo "KMeans" >> results_KM.txt
echo "Respiratory" >> results_KM.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_KM.txt
	python3 KMeans.py --dataset respiratory --n_clusters $j --log_interval 2 --n_epochs 50 >> results_KM.txt
done

echo "KMeans" >> results_KM.txt
echo "Wid_Mortality" >> results_KM.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_KM.txt
	python3 KMeans.py --dataset wid_mortality --n_clusters $j --log_interval 2 --n_epochs 50 >> results_KM.txt
done

echo "KMeans" >> results_KM.txt
echo "Diabetes" >> results_KM.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_KM.txt
	python3 KMeans.py --dataset diabetes --n_clusters $j --log_interval 2 --n_epochs 50 >> results_KM.txt
done