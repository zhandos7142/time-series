import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.graphics.tsaplots import plot_acf
from pmdarima.arima import ARIMA
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from sktime.forecasting.statsforecast import StatsForecastAutoETS
from sktime.forecasting.statsforecast import StatsForecastAutoTBATS
from sktime.forecasting.statsforecast import StatsForecastAutoTheta

warnings.filterwarnings("ignore")

def create_trend_X(bp, n):
    
    bp = sorted([0, n] + bp)
    X = np.ones((n, len(bp)))
    
    for i in range(len(bp)-1):
        start, end = bp[i], bp[i+1]
        X[:start, i+1] = 0
        X[start:end, i+1] = np.arange(1, end - start + 1)
        X[end:, i+1] = end - start
        
    return X

def create_dhr_X(T, P, N):
    
    t = np.arange(1, T+1)
    X = np.zeros((T, 2 * N))
    
    if P == 1:
        X = X
    else:
        for i in range(1, N + 1):
            X[:, 2*i-2] = np.sin(2 * np.pi * i * t / P)
            X[:, 2*i-1] = np.cos(2 * np.pi * i * t / P)
      
    return X

def ssr_n_general(data, n, j, h, prev_seg=None):
    
    ssr_n = []
    y = data[:n]  
    yty = y.T @ y

    for i in range(j * h, n - h + 1):
        if j == 0:
            bp_temp = []
        elif j == 1:
            bp_temp = [i]
        else:
            bp_temp = prev_seg[i - j * h] + [i]
        X = create_trend_X(bp_temp, n)
        XTX = X.T @ X
        XTy = X.T @ y
        L = np.linalg.cholesky(XTX)
        beta = np.linalg.solve(L.T, np.linalg.solve(L, XTy))
        ssr = yty - XTy.T @ beta
        ssr_n.append((ssr, bp_temp, beta))

    opt = min(ssr_n, key=lambda x: x[0])
    
    return opt[0], opt[1], opt[2] 

def compute_segmentation(data, j, h, prev_seg=None):
    
    seg_list = [] 
    
    for n in range((j + 1) * h, len(data) + 1):
        ssr, bps, beta = ssr_n_general(data, n, j, h, prev_seg)
        seg_list.append(bps)
    
    return (ssr, bps, beta), seg_list

def compute_multi_segmentation(data, max_num_bps, h, num_bps):
    
    opt_ssrs = []
    opt_bps = []
    opt_betas = []
    prev_seg = None
    
    if num_bps == None:
        for j in range(0, max_num_bps+1):
            optimal_part, seg_list = compute_segmentation(data, j, h, prev_seg)
            prev_seg = seg_list
            opt_ssrs.append(optimal_part[0])
            opt_bps.append(optimal_part[1])
            opt_betas.append(optimal_part[2])
        
        return opt_ssrs, opt_bps, opt_betas
        
    else:
        for j in range(0, num_bps+1):
            optimal_part, seg_list = compute_segmentation(data, j, h, prev_seg)
            prev_seg = seg_list
            
        return optimal_part
        
def detect_num_bps(metric, threshold, min_num_bps):
    
    pot_num_bps = min_num_bps
    
    for i in range(len(metric)):
        if metric[i]>threshold:
            pot_num_bps = min_num_bps + 1 + i
        else:
            break
            
    return pot_num_bps
        
def calc_metrics(train, test, forecast, season_length):
    
    mae = (np.sum(np.abs(forecast-test)))/len(test)
    rmse = np.sqrt((np.sum((forecast-test)**2))/len(test))
    smape = (100/len(test))*np.sum((np.abs(forecast-test))/((np.abs(forecast)+np.abs(test))/2))
    mase = mae/((np.sum(np.abs(train[season_length:] - train[:-season_length])))/(len(train)-season_length))
    
    return mae, rmse, smape, mase

h = 2
season_length = 4
p_max = 2
q_max = 2

df = [71.407, 73.597, 74.449, 75.726, 73.073, 75.075, 76.553, 78.731,
    76.313, 78.27, 79.127, 81.397, 78.871, 81.028, 82.412, 83.772,
    81.653, 83.963, 83.866, 85.682, 82.883, 85.344, 85.837, 87.817,
    85.305, 86.204, 85.506, 85.253, 81.109, 82.814, 83.68, 85.846,
    82.705, 85.431, 86.261, 88.039, 84.757, 86.718, 87.154, 89.166,
    87.289, 88.633, 89.265, 90.568, 88.114, 90.131, 91.504, 93.541,
    89.805, 92.44, 94.158, 96.055, 93.052, 95.794, 96.697, 97.886,
    94.673, 97.252, 98.29, 100.19, 95.826, 99.8, 100.976, 103.398,
    99.751, 102.793, 103.865, 105.457, 101.428, 105.227, 106.715, 109.137,
    102.997, 97.569, 104.73, 108.073, 104.698, 109.34, 110.309, 114.051,
    108.857, 111.855, 113.081, 115.62, 111.547, 115.159, 116.745, 118.938,
    114.999, 118.644, 119.658, 122.407, 117.244]

start = 2002
end = 2025
f = 4

plt.figure(figsize=(12, 12))
plt.subplot(2, 1, 1)
plt.plot(np.arange(start, end+1/season_length, 1/season_length), df, color='b')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Index', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.axvspan(end - (f-1)/season_length, end, color='grey', alpha=0.3)
plt.tight_layout()
plt.show()

df = np.log(df)
train = np.array(df[:-f])
test = np.array(df[-f:])
T = len(train)

plt.figure(figsize=(12, 12))
plt.subplot(2, 1, 1)
plt.plot(np.arange(start, start + T/season_length, 1/season_length), train, color='b')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Log(Index)', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.show()

max_num_bps = int(input('Maximum Number of Breaks: '))

ssrs, bps, betas = compute_multi_segmentation(train, max_num_bps, h, num_bps=None)
kneeL = ssr_n_general(np.array(ssrs), len(ssrs), 1, 2, prev_seg=None)
fitted_ssrs = create_trend_X(kneeL[1], len(ssrs)) @ kneeL[2]
min_num_bps = kneeL[1][0] - 1

filtered_ssrs = ssrs[min_num_bps:]
ssrs_filt_diff = []
for i in range(len(filtered_ssrs)-1):
    ssrs_filt_diff.append(filtered_ssrs[i]-filtered_ssrs[i+1])
diff_threshold = np.mean(ssrs_filt_diff)
pot_num_bps_diff = detect_num_bps(ssrs_filt_diff, diff_threshold, min_num_bps)
ssrs_ratio = []
for i in range(len(ssrs)-1):
    ssrs_ratio.append(1 - (ssrs[i+1]/ssrs[i]))
ratio_threshold = np.mean(ssrs_ratio)
pot_num_bps_ratio = detect_num_bps(ssrs_ratio, ratio_threshold, 0)

residuals = []
for i in range(len(bps)):
    
    bp = bps[i]
    beta = betas[i]
    X = create_trend_X(bp, T)
    trend_fitted = X @ beta
    resid = train - trend_fitted
    residuals.append(resid)
    
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(start, start + T/season_length, 1/season_length), train, color='b')
    plt.plot(np.arange(start, start + T/season_length, 1/season_length), trend_fitted, color='r')
    for j in range(len(bp)):
        plt.axvline(x = start + (bp[j]-1)/season_length, color='black', linestyle='--')
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(start, start + T/season_length, 1/season_length), resid, color='r')
    plt.axhline(y = 0, color = 'black')
    plt.subplot(3, 1, 3)
    plot_acf(resid, title='', bartlett_confint=False, ax=plt.gca(), auto_ylims=True)
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(24, 8))
plt.subplot(1, 3, 1)
plt.plot(np.arange(0, len(ssrs)), ssrs, marker='o', color='b')
plt.plot(np.arange(0, len(ssrs)), fitted_ssrs, color='g')
plt.xticks(np.arange(0, len(ssrs), step=2))
plt.axvline(x = min_num_bps, color='r', linestyle='--')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title('(a)', fontsize = 18)
plt.subplot(1, 3, 2)
plt.plot(np.arange(1, len(ssrs)), ssrs_ratio, marker='o', color='b')
plt.axhline(y = ratio_threshold, color='g')
plt.axvline(x = pot_num_bps_ratio, color='r', linestyle='--')
plt.xticks(np.arange(1, len(ssrs), step=2))
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title('(b)', fontsize = 18)
plt.subplot(1, 3, 3)
plt.plot(np.arange(min_num_bps+1, len(ssrs)), ssrs_filt_diff, marker='o', color='b')
plt.axhline(y = diff_threshold, color='g')
plt.axvline(x = pot_num_bps_diff, color='r', linestyle='--')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title('(c)', fontsize = 18)
plt.tight_layout()
plt.show()


opt_num_bps = int(input(f"Number of Breaks (0-{len(bps)-1}): ")) 
if season_length != 1:
    seasonality = input("Seasonality (True/False): ") == 'True'
else:
    seasonality = False
trend_resid = residuals[opt_num_bps]

orders_list = []
fitted_list = []
if seasonality == True:
    
    for k in range(1, int(season_length/2)+1):
        dhr_exog = create_dhr_X(T, season_length, k)
        for p in range(p_max+1):
            for q in range(q_max+1):
                try:
                    model = ARIMA(order = (p, 0, q), with_intercept = 'False', trend = 'n',
                                  mle_regression = 'True', enforce_stationarity = 'True',
                                  enforce_invertibility = 'True').fit(trend_resid, X = dhr_exog)
                    orders_list.append([k, p, q, model.aicc()])
                    fitted_list.append(model.fittedvalues())
                except Exception:
                    continue
    
else:
    for p in range(p_max+1):
        for q in range(q_max+1):
            try:
                model = ARIMA(order = (p, 0, q), with_intercept = 'False', trend = 'n',
                              mle_regression = 'True', enforce_stationarity = 'True',
                              enforce_invertibility = 'True').fit(trend_resid)
                orders_list.append([0, p, q, model.aicc()])
                fitted_list.append(model.fittedvalues())
            except Exception:
                continue
min_index = min(range(len(orders_list)), key=lambda i: orders_list[i][3])
str_order = orders_list[min_index][:3]
str_comp = train - fitted_list[min_index]

opt_bp = compute_multi_segmentation(str_comp, max_num_bps, h, num_bps = opt_num_bps)[1]
trend_exog = create_trend_X(opt_bp, T)
if seasonality == True:
    dhr_exog = create_dhr_X(T, season_length, str_order[0])
    exog = np.hstack([trend_exog, dhr_exog])
else:
    exog = trend_exog
model_ltsta = ARIMA(order = (str_order[1], 0, str_order[2]), with_intercept = 'False',
                       trend = 'n', mle_regression = 'True', enforce_stationarity = 'True',
                       enforce_invertibility = 'True').fit(train, X = exog)
print(model_ltsta.summary())
fig = plt.figure(figsize=(12, 10))
model_ltsta.plot_diagnostics(lags = 12, fig=fig)
plt.tight_layout()
plt.show()

fitted_ltsta = model_ltsta.fittedvalues()
trend_params = model_ltsta.params()[:opt_num_bps+2]
trend_fitted = trend_exog @ trend_params
if seasonality == True:
    dhr_params = model_ltsta.params()[opt_num_bps+2:opt_num_bps+2+2*str_order[0]]
    dhr_fitted = dhr_exog @ dhr_params
else:
    dhr_fitted = np.zeros(T)
arma_fitted = fitted_ltsta - (trend_fitted + dhr_fitted)
residuals = train - (trend_fitted + dhr_fitted + arma_fitted)

plt.figure(figsize=(12, 12))
plt.subplot(4, 1, 1)
plt.plot(np.arange(start, start + T/season_length, 1/season_length), trend_fitted, color='r')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title('Trend Component', fontsize = 14)
plt.subplot(4, 1, 2)
plt.plot(np.arange(start, start + T/season_length, 1/season_length), dhr_fitted, color='r')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title('Seasonality Component', fontsize = 14)
plt.subplot(4, 1, 3)
plt.plot(np.arange(start, start + T/season_length, 1/season_length), arma_fitted, color='r')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title('ARMA Component', fontsize = 14)
plt.subplot(4, 1, 4)
plt.plot(np.arange(start, start + T/season_length, 1/season_length), residuals, color='r')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title('Residual', fontsize = 14)
plt.tight_layout()
plt.show()

X_future = np.hstack([create_trend_X(opt_bp, T+f)[-f:], create_dhr_X(T+f, season_length, str_order[0])[-f:]])
forecast_ltsta = model_ltsta.predict(n_periods = f, X=X_future)
metrics_ltsta = calc_metrics(train, test, forecast_ltsta, season_length)

prediction_ltsta = np.concatenate((fitted_ltsta, forecast_ltsta))
plt.figure(figsize=(12, 12))
plt.subplot(2, 1, 1)   
plt.plot(np.arange(start, end+1/season_length, 1/season_length), df, color='blue')
plt.plot(np.arange(start, end+1/season_length, 1/season_length), prediction_ltsta, color='r', linestyle='--')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Log(Index)', fontsize=18)
plt.axvspan(end - (f-1)/season_length, end, color='grey', alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=14)
for i in range(len(opt_bp)):
    plt.axvline(x=start + (opt_bp[i]-1)/season_length, color='black', linestyle = '--')
plt.tight_layout()
plt.show()

model_arima = StatsForecastAutoARIMA(sp=season_length).fit(train)
forecast_arima = model_arima.predict(fh = np.arange(1, f+1)).flatten()
metrics_arima = calc_metrics(train, test, forecast_arima, season_length)

model_ets = StatsForecastAutoETS(season_length=season_length).fit(train)
forecast_ets = model_ets.predict(fh = np.arange(1, f+1)).flatten()
metrics_ets = calc_metrics(train, test, forecast_ets, season_length)

model_tbats = StatsForecastAutoTBATS(seasonal_periods = season_length).fit(train)
forecast_tbats = model_tbats.predict(fh = np.arange(1, f+1)).flatten()
metrics_tbats = calc_metrics(train, test, forecast_tbats, season_length)

model_theta = StatsForecastAutoTheta(season_length=season_length).fit(train)
forecast_theta = model_theta.predict(fh = np.arange(1, f+1)).flatten()
metrics_theta = calc_metrics(train, test, forecast_theta, season_length)

forecasts = [forecast_ltsta, forecast_arima, forecast_ets, forecast_theta, forecast_tbats]
models = ['LTSTA', 'AutoARIMA', 'AutoETS', 'AutoTBATS', 'AutoTheta']
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

plt.figure(figsize=(12, 12))
plt.subplot(2, 1, 1)
quarters = ['2024Q2', '2024Q3', '2024Q4', '2025Q1']
x_positions = np.arange(1, 5)
plt.plot(x_positions, test, color='black', marker='o', label = 'Actual Values')
for i in range(len(models)):
    plt.plot(x_positions, forecasts[i], color=colors[i], marker='o', label = models[i])
plt.xticks(x_positions, quarters)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel('Time', fontsize=18)
plt.ylabel('Log(Index)', fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
