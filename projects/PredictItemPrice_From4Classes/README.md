# Setup
I used python3.12 . Required packages: jupyter==1.1.1 , pandas==2.2.3 , seaborn==0.13.2 , scikit_learn==1.6.0 , openpyxl==3.1.5

To run: `jupyter notebook`

# Item Price Classification
Predict the price range of mobile phones based on 20 technical specifications(AKA features): RAM, battery power, screen size, etc. We will classify the phones into four price categories: low cost, medium cost, high cost, and very high cost.

The columns are battery_power,blue,clock_speed,dual_sim,fc,four_g,int_memory,m_dep,mobile_wt,n_cores,pc,px_height,px_width,ram,sc_h,sc_w,talk_time,three_g,touch_screen,wifi,price_range.

Here are 'few' columns with their descriptions:
* battery_power: Total energy a battery can store in one time measured in mAh
* blue: Has bluetooth or not
* dual_sim: Has dual sim support or not
* fc: Front Camera mega pixels
* four_g: Has 4G or not
* int_memory: Internal Memory in Gigabytes
* m_dep: Mobile Depth in cm
* pc: Primary Camera mega pixels
* sc_h: Screen Height of mobile in cm
* sc_w: Screen Width of mobile in cm
* talk_time: longest time that a single battery charge will last when you are talking

# Dataset
I downloaded the dataset from [kaggle] (https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?select=train.csv)

For detailed analysis, refer to the notebook: predict_price_range.ipynb
