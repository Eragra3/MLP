#regular train
#.\MLP.exe train --output mlp.json --sizes [70,200,10] --learning-rate 3 --momentum 0 --error-threshold 0 --max-epochs 200 --batch-size 10 --activation sigmoid --normal 0.7 --verbose

Start-Process -NoNewWindow -FilePath .\MLP.exe experiment --output lr/lr --values [0.5,1,2,3,4,5,6,7,10] --experiment learningrate --sizes [70,200,10] --momentum 0 --error-threshold 0 --max-epochs 200 --batch-size 10 --activation sigmoid --normal 0.7 --repetitions 1