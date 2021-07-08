import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_raw_visualization(data, feature_keys, title_units, colors):
    time_data = data.index
    fig, axes = plt.subplots(
        nrows=3, ncols=3, figsize=(12,12), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 3, i % 3],
            color=c,
            title="{} - {}".format(key, title_units[i]),
            rot=25,
        )
        ax.legend([feature_keys[i]])
    plt.tight_layout()
    
def niv_minmax_scaler(df_in, feature_range):
    feature_keys = list(df_in.columns)
    D=df_in
    D_shape = D.shape
    fmin, fmax = feature_range
    D_min = D.min(axis=0)
    D_datarange = (D.max(axis=0) - D.min(axis=0))
    D_std = (D - D.min(axis=0)) / D_datarange
    D_scaled = D_std * (fmax-fmin) + fmin
    dict_out = {'df_keys':feature_keys, 'df_shape':D_shape, 'feature_range':feature_range,'df_min':D_min,
                'df_datarange':D_datarange,'df_std':D_std,'df_scaled':D_scaled}
    return dict_out

def niv_minmax_inverse_scaler(df_in, dict_in):
    fmin = dict_in['feature_range'][0]
    fmax = dict_in['feature_range'][1]
    D_std_prime = (df_in - fmin) / (fmax-fmin)
    D_prime = D_std_prime * dict_in['df_datarange'] + dict_in['df_min']
    out = D_prime
    return out

def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()

# transform list into supervised learning format
def series_to_supervised(data, n_in, n_out=1):
	df = pd.DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = pd.concat(cols, axis=1)
	# drop rows with NaN values
	agg.dropna(inplace=True)
	return agg.values

def niv_dataset_split(data, y_col, split_train, split_val, split_test, past, future):
    index_train_split = int(split_train * int(data.shape[0]))
    index_val_split = index_train_split + int(split_val * int(data.shape[0]))
    index_test_split = index_val_split + int(split_test * int(data.shape[0])) - past - future

    y_start_train = past
    y_end_train = index_train_split + y_start_train + future -1

    y_start_val = index_train_split + past
    y_end_val = index_val_split + past + future - 1

    y_start_test = index_val_split + past
    y_end_test = index_test_split + past + future -1

    X_train = data.iloc[0:index_train_split,1:]
    y_train = data.iloc[y_start_train:y_end_train,y_col].values
    y_train = series_to_supervised(y_train, 0, n_out=future)

    X_val = data.iloc[index_train_split:index_val_split,1:]
    y_val = data.iloc[y_start_val:y_end_val,y_col].values
    y_val = series_to_supervised(y_val, 0, n_out=future)

    X_test = data.iloc[index_val_split:index_test_split,1:]
    y_test = data.iloc[y_start_test:y_end_test,y_col].values
    y_test = series_to_supervised(y_test, 0, n_out=future)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
def extract_y_indices(batches_in): 
    y_max = -10000
    y_min = 10000
    for batch_num, tup in enumerate(batches_in):
        x, y = tup
        y_avg = np.average(y, axis=1)
        y_maxtemp = np.amax(y_avg)
        y_mintemp = np.amin(y_avg)
        if y_maxtemp > y_max:
            y_max = y_maxtemp
            y_max_batch = batch_num
            y_max_ix = np.argmax(y_avg)
        if y_mintemp < y_min:
            y_min = y_mintemp
            y_min_batch = batch_num
            y_min_ix = np.argmin(y_avg)
    outmin = (y_min_batch, y_min_ix)
    outmax = (y_max_batch, y_max_ix)
    return outmin, outmax

def niv_forecast_series(model_in, batches_in, ix_batch, ix_series):
    batch=batches_in[ix_batch]
    batch_X = batch[0][ix_series]
    batch_X = np.reshape(batch_X, (1, batch_X.shape[0],batch_X.shape[1]))
    y_hat = model_in.predict(batch_X)[-1]
    y_data = batch[1][ix_series]
    batch_X =np.reshape(batch_X,(batch_X.shape[1],batch_X.shape[2]))
    return batch_X, y_data, y_hat

def niv_y_inverted(y, dict_in):
    y_key = dict_in['df_keys'][0]
    fmin = dict_in['feature_range'][0]
    fmax = dict_in['feature_range'][1]
    y_std_prime = (y - fmin) / (fmax-fmin)
    y_prime = y_std_prime * dict_in['df_datarange'][y_key] + dict_in['df_min'][y_key]
    out = pd.DataFrame(y_prime, columns=[y_key])
    return out

def niv_x_inverted(X, dict_in):
    X_keys = dict_in['df_keys'][1:]
    fmin = dict_in['feature_range'][0]
    fmax = dict_in['feature_range'][1]
    X_std_prime = pd.DataFrame((X - fmin) / (fmax-fmin), columns=X_keys)
    X_prime = X_std_prime * dict_in['df_datarange'][X_keys] + dict_in['df_min'][X_keys]
    out = X_prime
    return out

def niv_show_plot(df_X, y_data, y_hat, y_keys, y_colors, feature_keys, title):
    #fig, ax = plt.subplots(figsize=(8,6), dpi=100)
    
    ax = df_X.plot(x = 'time_steps', y=y_keys, color=y_colors, kind = 'line')

    ax.set_facecolor((1, 1, .99))
    ax.set_title(title, fontsize = 12, fontweight ='bold')

    ax.set_xlim(-df_X.shape[0]-25, y_data.shape[0])
   
    ax2=ax.twinx()
    df_X.plot(x = 'time_steps', y=['press'], kind = 'line', ax=ax2, color='tab:cyan', legend=False)

    kwargs1={'marker':'x', 'color':'r','label':'Test Data'}
    kwargs2={'marker':'o', 'color':'g','label':'Prediction'}

    y_data.plot(x = 'future_steps', y=feature_keys[0], kind = 'scatter', ax=ax, **kwargs1)
    y_hat.plot(x = 'future_steps', y=feature_keys[0], kind = 'scatter', ax=ax, **kwargs2)

    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Measurements', fontsize=12)
    ax2.set_ylabel('Pressure',fontsize=12, color='tab:cyan')

    ax.legend(loc='upper left', fontsize=10)

