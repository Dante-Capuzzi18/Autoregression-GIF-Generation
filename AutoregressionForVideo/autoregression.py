# This file is for fault detection using linear autoregression to predict the next n timesteps using the last p timesteps

import numpy as np

class Multimodal_AR():

    #will predict the next n timesteps using the last p timesteps from the given window
    #p = integer (use small, <10)
    #n = integer (use small, <10)
    #whole window set of timestamps from data collector, only of a single modality
    #will return the next n timestamps as a set like the window
    @staticmethod
    def AR(p, n, data):
        #making a copy of the window and checking that its at least p length
        data = np.asarray(data, dtype=float)
        l = len(data)
        if l <= p:
            raise ValueError(f"Need more than p={p} samples to fit model.")
    
        #creating a lag matrix and the target, then calculating coefficients to get bias and weights
        lag_mtrx = np.column_stack([data[i:l - p + i] for i in range(p)][::-1])
        target = data[p:]
        lag_mtrx = np.column_stack([np.ones(len(lag_mtrx)), lag_mtrx])
        coeffs, *_ = np.linalg.lstsq(lag_mtrx, target, rcond=None)
        bias, weights = coeffs[0], coeffs[1:]
        
        #predicting each next step n by using earlier predictions to predict later ones
        history = list(data[-p:])
        preds = []
        for _ in range(n):
            next_val = bias + np.dot(weights, history[::-1])
            preds.append(next_val)
            history.append(next_val)
            history.pop(0)
        
        #returning the array of predictions
        return np.array(preds)
    
    #will predict the next window up to n using p last steps, returning all modalities as a window copy
    @staticmethod
    def predict_next_window(p, n, window):
        #get a list of modalities
        modalities = list(window[0].keys())

        #build arrays per modality
        modality_series = {m: np.array([w[m] for w in window]) for m in modalities}

        #run AR prediction for each modality
        modality_preds = {m: Multimodal_AR.AR(p, n, modality_series[m]) for m in modalities}

        #combine into list of dicts
        prediction_window = [
            {m: modality_preds[m][t] for m in modalities} for t in range(n)
        ]

        return prediction_window

#TEST
print(" - AR Test - ")
window = [
    #constant, 2^n series, constant increasing, constant decreasing
    {'1': 1, '2': 1, '3': 1, '4': -1, }, #t=1
    {'1': 1, '2': 2, '3': 2, '4': -2, }, #t=2
    {'1': 1, '2': 4, '3': 3, '4': -3, }, #t=3
    {'1': 1, '2': 8, '3': 4, '4': -4, }, #t=4
    {'1': 1, '2': 16, '3': 5, '4': -5, }, #t=5
    {'1': 1, '2': 32, '3': 6, '4': -6, }, #t=6
    {'1': 1, '2': 64, '3': 7, '4': -7, }, #t=7
    {'1': 1, '2': 128, '3': 8, '4': -8, }, #t=8
    {'1': 1, '2': 256, '3': 9, '4': -9, }, #t=9
    {'1': 1, '2': 512, '3': 10, '4': -10, }, #t=10
]
preds = Multimodal_AR.predict_next_window(p=5, n=3, window=window)
i=1
for step in preds:
    print(f"{i}: {step}")
    i += 1