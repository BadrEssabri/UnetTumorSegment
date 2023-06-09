# UnetTumorSegment

## setup vm
# Download model and data
1. Download cuda and set the sourcing in the bashrc
```
https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local
export PATH="/usr/local/cuda-12.1/bin:$PATH"
nvcc --version
sudo prime-select nvidia
prime-select query
```
2. clone github repo (setup ssh keys):
```
connect the ssh keys
git clone git@github.com:BadrEssabri/UnetTumorSegment.git
```
2. Download required packages:
dont use pip install -r requirements.txt
```
pip install torch torchvision torchaudio
pip install h5py matplotlib tqdm 

```
3. Download dataset from Google Drive (train_brain and val_brain):
```
pip install gdown
cd UnetTumorSegment/data/train_brain
gdown --id 1_ePoCdcF6qOVp5SDPRffG7nwMLjvUaJm
unzip train_brain.zip
rm -rf train_brain.zip
gdown --id 13KCXTn1_fD9f-oLq9FIf5e8LeaUgJjot
unzip val_brain.zip
rm -rf val_brain.zip

```
# Test the program
1. Check all the paths in the python program
2. Run the program 
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
