import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dbpn_df = pd.read_csv('/home/ser606/ZhenLi/super_resolution/experiments/DBPN_in3f64b4_x4/results/train_results.csv')
ddbpn_df = pd.read_csv('/home/ser606/ZhenLi/super_resolution/experiments/D-DBPN_in3f64b4_x4/results/train_results.csv')

plt.plot(dbpn_df.psnr, 'r--', dbpn_df.psnr, 'b--')
plt.show()

plt.plot(drbpn_df.ssim, 'r--', ddbpn_df.ssim, 'b--')
plt.show()

