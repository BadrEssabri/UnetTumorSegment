# UnetTumorSegment

## setup vm
# Download model and data
1. Clone github repo (maybe setup ssh keys, dont know what already is installed)
```
sudo apt-get install python3.8
(sudo) apt install python3-pip
git clone
pip install gdown
```
2. Download dataset from Google Drive (train_brain and val_brain):
```
pip install gdown
cd UnetTumorSegment/data/train_brain
gdown https://drive.google.com/file/d/1_ePoCdcF6qOVp5SDPRffG7nwMLjvUaJm/view
unzip -q train_brain.zip
rm -rf train_brain.zip
gdown https://drive.google.com/file/d/13KCXTn1_fD9f-oLq9FIf5e8LeaUgJjot/view
unzip -q val_brain.zip
rm -rf val_brain.zip

```
# Setup CUDA
1. Check CUDA version on the top right of the given output (probably 11.6)
```
nvidia-smi
```
2. Setup torch to use this version of cuda (this could be different for other cuda versions, see https://discuss.pytorch.org/t/problem-with-cuda-11-6-installation/156119)
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```
# Test the program
1. Check all the paths in the python program
2. Run the program (if correct conda environment is activated)
```
python3 ./train.py
```
3. Check if no errors occur
# Run the program as background process
To run the program, even if you closed the VM, you need to run the script as a background process and save the outputs in a text file.
1. Get location of python version
```commandline
which python
```
2. Start the background process
```commandline
nohup <absolute path to anaconda python> -u <absolute path to python script> > outputfile.txt &
```
3. Check if the outputfile.txt contains the output of the program, if not something went wrong
```commandline
cat outputfile.txt
```
4. Check if the process is running in the background, you should see multiple running process
```commandline
ps aux | grep <Name of running script>
```
5. If you want to check the status of your program at a later time
```commandline
cat outputfile.txt
```

# Download results
If the program is finished, download the results (weights of the model)
1. ZIP the output folder (if line is not working, delete -q)
```commandline
zip -r -q weights.zip ....
```
2. Download the results.zip from the VM