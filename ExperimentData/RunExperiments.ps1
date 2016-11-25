#regular train
#.\MLP.exe train --output mlp.json --sizes [70,200,10] --learning-rate 7 --momentum 0 --error-threshold 0 --max-epochs 300 --batch-size 20 --activation sigmoid --normal 0.3 --verbose

#Start-Process -NoNewWindow -FilePath .\MLP.exe -ArgumentList "experiment --output lr/lr --values [0.5,1,2,3,4,5,6,7,10] --experiment learningrate --sizes [70,200,10] --momentum 0 --error-threshold 0 --max-epochs 200 --batch-size 10 --activation sigmoid --normal 0.7 --repetitions 3"

#Start-Process -NoNewWindow -FilePath .\MLP.exe -ArgumentList "experiment --output ndsd/ndsd --values [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] --experiment standarddeviation --sizes [70,200,10] --learning-rate 4 --momentum 0 --error-threshold 0 --max-epochs 200 --batch-size 10 --activation sigmoid --normal 0.7 --repetitions 3"

##activation functions

#Start-Process -NoNewWindow -FilePath .\MLP.exe -ArgumentList "experiment --output af/af_lr-0.25 --values [0,1] --experiment activationfunction --sizes [70,200,10] --learning-rate 0.25 --momentum 0 --error-threshold 0 --max-epochs 200 --batch-size 10 --activation sigmoid --normal 0.3 --repetitions 3"

##

## tanh

#Start-Process -NoNewWindow -FilePath .\MLP.exe -ArgumentList "experiment --output lr_tanh/lr --values [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8] --experiment learningrate --sizes [70,200,10] --momentum 0 --error-threshold 0 --max-epochs 200 --batch-size 10 --activation tanh --normal 0.1 --repetitions 3"

#Start-Process -NoNewWindow -FilePath .\MLP.exe -ArgumentList "experiment --output ndsd_tanh/ndsd --values [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15] --experiment standarddeviation --sizes [70,200,10] --learning-rate 0.1 --momentum 0 --error-threshold 0 --max-epochs 200 --batch-size 10 --activation tanh --repetitions 3"

## BEST NN

#.\MLP.exe train --output mlp_sigmoid.json --sizes [70,200,10] --learning-rate 4 --momentum 0 --error-threshold 0 --max-epochs 400 --batch-size 10 --activation sigmoid --normal 0.4

#.\MLP.exe train --output mlp_tanh.json --sizes [70,200,10] --learning-rate 0.6 --momentum 0 --error-threshold 0 --max-epochs 400 --batch-size 10 --activation tanh --normal 0.09

#normalized
#.\MLP.exe train --output mlp_sigmoid.json --sizes [70,200,10] --learning-rate 4 --momentum 0 --error-threshold 0 --max-epochs 400 --batch-size 10 --activation sigmoid --normal 0.4 -n

.\MLP.exe train --output mlp_tanh.json --sizes [70,200,10] --learning-rate 0.6 --momentum 0 --error-threshold 0 --max-epochs 400 --batch-size 10 --activation tanh --normal 0.09 -n