@echo off
REM run_train_full_args.bat
REM Batch script with populated fields for DeepFit training on Windows CMD.

:: Pre-populated configuration
set TrainRoot=D:\PUL - DeepFit\Dresscode
set ValRoot=D:\PUL - DeepFit\Test
set Categories=dresses upper_body lower_body upper_body1

set Device=cuda
set BatchSize=2
set EffectiveBatchSize=128
set NumEpochs=79
set Lr=1e-5
set Seed=42

set WandbProject=your_wandb_project
set WandbEntity=your_wandb_entity
:: Generate WandB run name without spaces (user can adjust as needed)
set WandbName=run_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
:: Replace spaces in time (e.g., leading space in hour) with zero
set WandbName=%WandbName: =0%
:: Remove any colons from time
set WandbName=%WandbName::=%

set CheckpointDir=E:\Deepfit-Training\checkpoints
set SaveEverySteps=100
set ResumeStep=
set DebugFlag=--debug

:: Ensure checkpoint directory exists
if not exist "%CheckpointDir%" (
    mkdir "%CheckpointDir%"
)

:: Construct python command
set CMD=python train.py ^
  --train_root "%TrainRoot%" ^
  --val_root "%ValRoot%" ^
  --categories %Categories% ^
  --device %Device% ^
  --batch_size %BatchSize% ^
  --effective_batch_size %EffectiveBatchSize% ^
  --num_epochs %NumEpochs% ^
  --lr %Lr% ^
  --seed %Seed%

if not "%WandbProject%"=="" (
    set CMD=%CMD% --wandb_project %WandbProject%
    if not "%WandbEntity%"=="" set CMD=%CMD% --wandb_entity %WandbEntity%
    if not "%WandbName%"=="" set CMD=%CMD% --wandb_name %WandbName%
)

set CMD=%CMD% --checkpoint_dir "%CheckpointDir%" --save_every_steps %SaveEverySteps%
if not "%ResumeStep%"=="" set CMD=%CMD% --resume_step %ResumeStep%
if defined DebugFlag set CMD=%CMD% %DebugFlag%

:: Execute
echo Running: %CMD%
%CMD%
