import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# load the data
df = pd.read_csv(r"C:\Users\Joshi Sir\Downloads\SDSS_quasar.dat", delim_whitespace=True)

df['Radio'].replace(-1.0, np.nan, inplace=True)
df['X-ray'].replace(-9.0, np.nan, inplace=True)


#color color pair plots

sns.pairplot(df[['u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag']])

plt.suptitle("Paired ugriz Magnitude Plots", y=1.02)
plt.show()

# finding outliers based on i_mag values
outliers = df[df['i_mag'] > 19]
print(f"Number of outliers with i_mag > 19: {len(outliers)}")

#predicting redshift

#df['u-g'] = df['u_mag'] - df['g_mag']
#df['g-r'] = df['g_mag'] - df['r_mag']
#df['r-i'] = df['r_mag'] - df['i_mag']
#df['i-z'] = df['i_mag'] - df['z_mag']

#features = ['u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag', 'u-g', 'g-r', 'r-i', 'i-z']
#X = df[features]
#y = df['z']

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#reg = RandomForestRegressor(n_estimators=100)
#reg.fit(X_train, y_train)
#y_pred = reg.predict(X_test)

#rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#print(f"redshift prediction (rmse): {rmse:.4f}")

#xray/optical , xray/radio ratios

#if 'X-ray' in df.columns and 'Radio' in df.columns:
#    df['X_opt_ratio'] = df['X-ray'] / df['i_mag']
#    df['X_radio_ratio'] = df['X-ray'] / df['Radio']

#    plt.scatter(df['z'], df['X_opt_ratio'], alpha=0.5)
#    plt.xlabel('Redshift')
#    plt.ylabel('X-ray / i_mag')
#    plt.title('X-ray/Optical Ratio vs Redshift')
#    plt.show()

#    plt.scatter(df['z'], df['X_radio_ratio'], alpha=0.5)
#    plt.xlabel('Redshift')
#    plt.ylabel('X-ray / Radio')
#    plt.title('X-ray/Radio Ratio vs Redshift')
#    plt.show()

#photometric radio loudness prediction

df['is_radio_loud'] = df['Radio'] > 0

X = df[features]
y = df['is_radio_loud'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_radio = RandomForestRegressor(n_estimators=100)
rf_radio.fit(X_train, y_train)
radio_preds = rf_radio.predict(X_test)
radio_preds = (radio_preds > 0.5).astype(int)

accuracy = np.mean(radio_preds == y_test)
print(f"Radio-loudness Prediction Accuracy: {accuracy:.2f}")

#analysing high redishift qusars with redshift lower bound of 4

high_z = df[df['z'] > 4]
plt.scatter(df['u-g'], df['g-r'], alpha=0.3, label='All')
plt.scatter(high_z['u-g'], high_z['g-r'], color='red', label='z > 4')
plt.xlabel('u - g')
plt.ylabel('g - r')
plt.title('High Z quasars in color-color plot')
plt.legend()
plt.show()

# simple demonstration of lyman alpha forests in high redshift quasars

colors = ['u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag']
for color in colors:
    plt.scatter(df['z'], df[color], s=2, label=color)
plt.xlabel("Redshift z")
plt.ylabel("Magnitude")
plt.title("Lyman-alpha Forest Effect")
plt.legend()
plt.show()

#relationship of xray and i_mag

#if 'X-ray' in df.columns:
#    df['X_minus_i'] = df['X-ray'] - df['i_mag']
#    plt.scatter(df['z'], df['X_minus_i'], alpha=0.5)
#    plt.xlabel('Redshift')
#    plt.ylabel('X-ray - i_mag')
#    plt.title('X-ray - i_mag vs Redshift')
#    plt.show()


#plotting absolute magnitudes

#plt.hist(df['M_i'], bins=50, color='teal')
#plt.xlabel("M_i")
#plt.ylabel("Count")
#plt.title("Absolute Magnitude M_i Distribution")
#plt.show()

