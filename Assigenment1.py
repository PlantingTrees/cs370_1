import numpy as np 

from scipy.stats import  norm, poisson
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#need 4 varibles
# ak = air knots ={p dist. correlated}
#pp = passed cut point ={p dist. correlated}
#ptime= practise time ={cont dist. correlated}
#gs=glove size (5.5 -9) + o.5 increments {={p dist. not correlated}}

#correlation range = [-1, 1]
# (ak, pp) = 0.6
#(ak, ptime)= -0.9
#(pp, ptime) = -0.5

#makee the coveriance matrix
Sigma= np.array([[1, 0.6, -0.9], [0.6, 1, -0.5], [-0.9, -0.5, 1]])
mean=[0, 0, 0]
#mvnorm fucn
def MVNormal(n=1, mu=mean, Sigma=None):
    if Sigma is None:
        raise ValueError("Covariant matrix is not initialized yet, do it please.")
    samples= np.random.multivariate_normal(mu,Sigma,n)
    return samples

#set seed
np.random.seed(42)
no_samples=int(1e4)
APT_distributions= MVNormal(no_samples, Sigma=Sigma)
correlations_matrix= np.corrcoef(APT_distributions,rowvar=False)
#print(correlations_matrix)

#plotting correlations

var_labels=["ak", "pp", "ptime"] 
df= pd.DataFrame(APT_distributions,columns=var_labels)
sns.set(style="ticks")
sns.pairplot(df, diag_kind="kde", markers="o")
plt.savefig("corr_plot.png")
std_mean=0
std_dev=1
#perfoming thr PIT
std_normal_cdf= norm.cdf(APT_distributions, loc=std_mean, scale=std_dev)
correlation_cdf = np.corrcoef(std_normal_cdf, rowvar=False)
print(correlation_cdf)
var_labels=["ak", "pp", "ptime"] 
df= pd.DataFrame(std_normal_cdf,columns=var_labels)
sns.set(style="ticks")
sns.pairplot(df, diag_kind="kde", markers="o")
plt.savefig("corr_plot_cdf.png")

print(len(correlation_cdf))

#making inverse transform for each column labels
ak_lambda_val =5 
ak= poisson.ppf(std_normal_cdf[:, 0], ak_lambda_val)

pp_mean=15
pp=poisson.ppf(std_normal_cdf[:, 1], pp_mean)

ptime_mean=120
p_time_sd=30
ptime=norm.ppf(std_normal_cdf[:,2], loc=ptime_mean, scale=p_time_sd)

#non correlated gs
mean_gs=7.25
std_dev_gs=0.875
gs=np.random.normal(loc=mean_gs, scale=std_dev_gs, size=no_samples)



#make a dataframe for the variables
data= pd.DataFrame({'ak':ak, 'pp':pp, 'ptime':ptime, 'gs':gs})
sns.set(style="ticks")
sns.pairplot(data, diag_kind="kde", markers="o")
plt.savefig("final_result.png")


#PART 2
# Initialize variables for the running statistics
#using this formula 
#R_i = (1 / (i-1)) * Σ [x_t * x_t'] - μ_i * μ_i'
n = 0
mean = 0
M2 = 0
sample_corr_list = []  # List to store sample correlation matrices

num_samples = 100
#rnadom data 
stream = np.random.randn(num_samples, 4)

for i in range(num_samples):
    x = stream[i]  # get from random stream data
    
    n += 1
    delta = x - mean #change in mean
    mean += delta / n
    delta2 = x - mean
    M2 += delta * delta2

    if i == 0:
        R = np.outer(x, x)
    else:
        R = ((i - 1) * R + np.outer(x, x)) / i

    if i >= 1:
        sample_corr_list.append(R / (np.outer(np.sqrt(np.diag(R)), np.sqrt(np.diag(R)))))

# Plot the elements of the sample correlation matrix
for i, R_i in enumerate(sample_corr_list):
    plt.figure()
    plt.imshow(R_i, cmap='coolwarm', vmin=-1, vmax=1, aspect='equal')
    plt.title(f'Online Setting')
    plt.colorbar()
    plt.savefig("problem 2.png")