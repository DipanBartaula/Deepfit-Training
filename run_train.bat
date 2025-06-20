@echo off
REM run_train_full_args.bat
REM Batch script with populated fields for DeepFit training on Windows CMD.

:: --- Configuration ---
set TrainRoot=D:\PUL - DeepFit\Dresscode
set ValRoot=D:\PUL - DeepFit\Test
set Categories=dresses upper_body lower_body upper_body1

set Device=cuda
set BatchSize=2
set EffectiveBatchSize=128
set NumEpochs=79
set Lr=1e-5
set Seed=42

set WandbProject=Deepfit_Training
set WandbEntity=mannfeyn71-tribhuvan-university-institute-of-engineering

:: Generate WandB run name (YYYYMMDD_HHMMSS)
for /f "tokens=2 delims==" %%I in ('wmic OS GET LocalDateTime /value') do set ldt=%%I
set WandbName=run_%ldt:~0,8%_%ldt:~8,6%

set CheckpointDir=E:\Deepfit-Training\checkpoints
set SaveEverySteps=100
set ResumeStep=
set DebugFlag=--debug

:: --- Prep ---
if not exist "%CheckpointDir%" (
    mkdir "%CheckpointDir%"
)

:: Ensure WANDB_API_KEY is in the environment
:: (you can also `set WANDB_API_KEY=your_key_here` here)
if "%WANDB_API_KEY%"=="" (
    echo WARNING: WANDB_API_KEY not set in environment.
)

:: --- Build command ---
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
    if not "%WandbName%"==""   set CMD=%CMD% --wandb_name %WandbName%
)

set CMD=%CMD% --checkpoint_dir "%CheckpointDir%" --save_every_steps %SaveEverySteps%
if not "%ResumeStep%"=="" set CMD=%CMD% --resume_step %ResumeStep%
if defined DebugFlag    set CMD=%CMD% %DebugFlag%

:: --- Execute ---
echo.
echo Running:
echo    %CMD%
echo.

%CMD%
