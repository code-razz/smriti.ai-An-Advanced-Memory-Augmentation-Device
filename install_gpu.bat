@echo off
echo Uninstalling CPU versions...
.\svenv\Scripts\pip uninstall -y torch torchaudio torchvision

echo Installing CUDA versions (This may take a while)...
.\svenv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Done!
