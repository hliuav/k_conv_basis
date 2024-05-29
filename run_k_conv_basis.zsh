# list of k values of python input parameters

#k_values=(5 10 20 40 80 160 320 640 1280 2560 5120 10240)
k_values=(5 10 20 40 80 160) 320 640)
#k_values=(640 1280 2560 5120 10240)
#k_values=(20480 32000)

# loop over k values
for k in ${k_values[@]}
do
    python k_conv_basis.py $k
    echo "k = $k"
done
