import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


np.random.seed(42)
n_rows = 10000


data = pd.DataFrame({
    'driver_age': np.random.randint(18, 75, n_rows),
    'vehicle_value': np.random.randint(5000, 50000, n_rows),
    'miles_driven': np.random.randint(2000, 20000, n_rows)
})


data['age_segment'] = pd.cut(data['driver_age'], bins=[17, 25, 50, 75], labels=['Young', 'Middle', 'Senior'])

data = pd.get_dummies(data, columns=['age_segment'], drop_first=False, dtype=int)





lambda_freq = np.exp(-3.5 + 0.5 * data['age_segment_Young'] + 0.00005 * data['miles_driven'])
data['claim_count'] = np.random.poisson(lambda_freq)



mu_sev = 500 + 0.05 * data['vehicle_value']

scale = mu_sev / 2.0
data['claim_cost'] = data['claim_count'] * np.random.gamma(shape=2.0, scale=scale)





freq_cols = ['age_segment_Young', 'miles_driven']
X_freq = sm.add_constant(data[freq_cols])
freq_model = sm.GLM(data['claim_count'], X_freq, family=sm.families.Poisson()).fit()



severity_data = data[data['claim_count'] > 0].copy()
severity_data['avg_claim_cost'] = severity_data['claim_cost'] / severity_data['claim_count']

X_sev = sm.add_constant(severity_data[['vehicle_value']])
sev_model = sm.GLM(severity_data['avg_claim_cost'], X_sev, family=sm.families.Gamma(link=sm.families.links.log())).fit()




data['pred_frequency'] = freq_model.predict(X_freq)

X_sev_predict = sm.add_constant(data[['vehicle_value']]) 
data['pred_severity'] = sev_model.predict(X_sev_predict)

data['pure_premium'] = data['pred_frequency'] * data['pred_severity']

print("--- FREQUENCY MODEL SUMMARY (Poisson) ---")
print(freq_model.summary())
print("\n--- SEVERITY MODEL SUMMARY (Gamma) ---")
print(sev_model.summary())

print("\n--- SAMPLE RATEMAKING OUTPUT ---")
print(data[['driver_age', 'vehicle_value', 'pred_frequency', 'pred_severity', 'pure_premium']].head())

plt.scatter(data['driver_age'], data['pure_premium'], alpha=0.5)
plt.title("Actuarial Pure Premium by Driver Age")
plt.xlabel("Age")
plt.ylabel("Calculated Premium ($)")
plt.show()
