:: run_train.bat
@echo off
REM Usage: run_train.bat <micro_batch_size> <effective_batch_size>
REM Gradient checkpointing is enabled by default in train.py
if "%1"=="" (
    echo Please provide micro batch size and effective batch size.
    echo Usage: run_train.bat 4 16
    goto :eof
)
set MICRO_BATCH=%1
set EFFECTIVE_BATCH=%2
python train.py --batch_size %MICRO_BATCH% --effective_batch_size %EFFECTIVE_BATCH% --epochs 10 --lr 1e-4 --num_workers 4 --output_dir checkpoints --pretrained "stabilityai/stable-diffusion-2-inpainting" --max_grad_norm 1.0
echo DeepFit training launched with micro batch %MICRO_BATCH%, effective batch %EFFECTIVE_BATCH% (gradient checkpointing enabled by default)
