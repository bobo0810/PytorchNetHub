python3 ../train.py \
--train-dir ../../data/train_4200 --train-size 2000 \
--valid-dir ../../data/valid_800 --valid-size 400 \
--ckpt-overwrite \
--ckpt-save-path ../../ckpts \
--report-interval 100 \
--nb-epochs 100 \
--loss l2 \
--noise-type gaussian \
--noise-param 50 \
--crop-size 128 \
--plot-stats \
--cuda

python3 ../train.py \
  --train-dir ../../data/train_4200 --train-size 2000 \
  --valid-dir ../../data/valid_800 --valid-size 400 \
  --ckpt-overwrite \
  --ckpt-save-path ../../ckpts \
  --report-interval 100 \
  --nb-epochs 100 \
  --loss l2 \
  --noise-type gaussian \
  --noise-param 50 \
  --crop-size 128 \
  --clean-targets \
  --plot-stats \
  --cuda
  
python3 ../train.py \
  --train-dir ../../data/train_4200 --train-size 2000 \
  --valid-dir ../../data/valid_800 --valid-size 400 \
  --ckpt-overwrite \
  --ckpt-save-path ../../ckpts \
  --report-interval 100 \
  --nb-epochs 100 \
  --loss l1 \
  --noise-type text \
  --noise-param 0.5 \
  --crop-size 128 \
  --plot-stats \
  --cuda
  
python3 ../train.py \
  --train-dir ../../data/train_4200 --train-size 2000 \
  --valid-dir ../../data/valid_800 --valid-size 400 \
  --ckpt-overwrite \
  --ckpt-save-path ../../ckpts \
  --report-interval 100 \
  --nb-epochs 100 \
  --loss l1 \
  --noise-type text \
  --noise-param 0.5 \
  --crop-size 128 \
  --clean-targets \
  --plot-stats \
  --cuda