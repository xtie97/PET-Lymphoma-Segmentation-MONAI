import os 
from monai.apps.auto3dseg import AutoRunner
#os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/requirements-dev.txt") 
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install fire") 
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install numpy==1.23.4") 

# the minimum required code is to create an AutoRunner() and call runner.run()
# the algos must be set to 'segresnet' (since currently it's the only algo with support of multi-resolution input images, such as CT and PET)
# here we also set ensemble=False (optional) to prevent inference on the testing set (since we do not use any testing sets, only the 5-fold cross validation)
# for you own inference (and ensemble) you can provide a list of testing files in "hecktor22_folds.json"
runner = AutoRunner(input="input.yaml", algos="segresnet", work_dir="./work_dir", train=True, ensemble=False)
runner.set_num_fold(num_fold=1) #4/5=0.8, 5/6=0.83
## optionally, we can use just 1-fold (for a quick training of a single model, instead of training 5 folds)
train_param = {
                "CUDA_VISIBLE_DEVICES": [0],
                "num_iterations": 200,
                "num_iterations_per_validation": 50,
                "num_images_per_batch": 3,
                "num_epochs": 150,
            }
runner.set_training_params(train_param)
runner.run()
# 3 images per batch, 192*192*192 fit the model (30G GPU memory)
## optionally, we can define the path to the dataset here, instead of the one in input.yaml
# runner.set_training_params({"dataroot" : '/data/hecktor22'})



