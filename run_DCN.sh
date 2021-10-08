echo "DCN" >> results_DCN.txt
echo "CIC" >> results_DCN.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_DCN.txt
	python3 DCN.py --dataset cic --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50
done

echo "DCN" >> results_DCN.txt
echo "Sepsis" >> results_DCN.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_DCN.txt
	python3 DCN.py --dataset sepsis --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50
done

echo "DCN" >> results_DCN.txt
echo "Kidney" >> results_DCN.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_DCN.txt
	python3 DCN.py --dataset kidney --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50
done

echo "DCN" >> results_DCN.txt
echo "Respiratory" >> results_DCN.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_DCN.txt
	python3 DCN.py --dataset respiratory --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50
done

echo "DCN" >> results_DCN.txt
echo "Wid_Mortality" >> results_DCN.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_DCN.txt
	python3 DCN.py --dataset wid_mortality --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50
done

echo "DCN" >> results_DCN.txt
echo "Diabetes" >> results_DCN.txt

for j in 1 2 3 4
do
	echo "k = $(($j))" >> results_DCN.txt
	python3 DCN.py --dataset diabetes --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50
done