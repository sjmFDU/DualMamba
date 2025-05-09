import spectral
import os
import scipy.io as io

def plot_spectral(dataset_name, dataset_dir='./datasets'):
    image = None
    if (dataset_name == 'whulk'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_LongKou.mat"))
        image = image['WHU_Hi_LongKou']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_LongKou_gt.mat"))
        gt = gt['WHU_Hi_LongKou_gt']
        labels = [
            'Undefined',
            'Corn',
            'Cotton',
            'Sesame',
            'Broad-leaf soybean',
            'Narrow-leaf soybean',
            'Rice',
            'Water',
            'Roads and houses',
            'Mixed weed',
        ]
        rgb_bands = [240, 200, 140]  # to be edited


    elif (dataset_name == 'ip'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, 'Indian_pines_corrected.mat'))
        image = image['indian_pines_corrected']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, 'Indian_pines_gt.mat'))
        gt = gt['indian_pines_gt']
        labels = ['Undefined', 'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
                'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                'Stone-Steel-Towers']
        rgb_bands = [29, 19, 9]
        
    elif (dataset_name == 'hu2018'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, 'Houston2018.mat'))
        image = image['houston2018']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, 'Houston2018_gt.mat'))
        gt = gt['houston2018_gt']
        labels = ['Undefined', 'Healthy Grass', 'Stressed Grass', 'Artificial turf', 
                  'Evergreen trees', 'Deciduous trees', 'Bare earth', 
                  'Water', 'Residential buildings', 'Non-residential buildings', 
                  'Roads', 'Sidewalks', 'Crosswalks', 'Major thoroughfares', 
                  'Highways', 'Railways', 'Paved parking lots', 
                  'Unpaved parking lots', 'Cars', 'Trains', 'Stadium seats']
        rgb_bands = [40,32,24]
        
    spectral.save_rgb('./rgbimage/' + f'{dataset_name}_{rgb_bands[0]}_{rgb_bands[1]}_{rgb_bands[2]}.jpg', image, rgb_bands)

plot_spectral('whulk')