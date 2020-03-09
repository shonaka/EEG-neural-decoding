import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy import signal
from scipy import stats
import pdb


def validate_results(joints, seg_len, actual, pred):
    '''A wrapper function to validate the results using various measurements.

    '''

    # 1) MSE
    mse_val = mean_squared_error(actual, pred)

    # 2) R2
    r2_val = r2_score(actual, pred)

    # 3) R-value segments
    median_rvals = {}  # for each joint
    for i, j in enumerate(joints):
        median_rvals[j] = calculate_rvals(seg_len,
                                          actual[:, i],
                                          pred[:, i])

    # Put the results into a dictionary
    results = {}
    results['mse'] = mse_val
    results['r2'] = r2_val
    results['median_rval'] = median_rvals

    return results

def calculate_rvals(seglen, actual, pred):
    '''Calculate r-values per segment.

    Given the segment length "seglen", calculate the r-values per segment per joint.

    Arguments:
        seglen: segment length.
        actual: actual measurement of a joint angle.
        pred:   predicted a joint angle.

    Returns:
        median_rval:    median r-value
    '''

    # Just checking
    assert type(
        seglen) == int, "Segment length has to be integers. (e.g. 200 samples = 200)"

    # Initialize variables
    rval = []

    # But for now, just segment into segment length
    len_samp = len(actual)
    num_chunk = len_samp // seglen
    for i in range(num_chunk):
        act_chunk = actual[seglen*i: seglen*(i+1)]
        pre_chunk = pred[seglen*i: seglen*(i+1)]
        tmp = np.corrcoef(act_chunk, pre_chunk)[0, 1]
        rval.append(tmp)

    # calculate the median r-value
    median_rval = np.median(rval)

    return median_rval

def calc_rval(ind_act, ind_pred, joints):
    '''Calculate r-value and store them in a dict for each joint

    Arguments:
        ind_act:    a list of actual joint angle values [num samples x num joints]
        ind_pred:   a list of predicted joint angles [num samples x num joints]
        joints:     a dictionary with joint names {0:'Hip', 1:'Knee', 2:'Ankle'}

    Returns:
        r_vals:     a dictionary containing r-value for each joint
    '''

    r_vals = {}
    for k, j in joints.items():
        r_vals[j] = np.corrcoef(ind_act[:, k], ind_pred[:, k])[0, 1]
    return r_vals

def calc_r2(ind_act, ind_pred, joints):
    '''Calculate r2 score and store them in a dict for each joint

    Arguments:
        ind_act:    a list of actual joint angle values [num samples x num joints]
        ind_pred:   a list of predicted joint angles [num samples x num joints]
        joints:     a dictionary with joint names {0:'Hip', 1:'Knee', 2:'Ankle'}

    Returns:
        r2_vals:     a dictionary containing r2 score for each joint
    '''

    r2_vals = {}
    for k, j in joints.items():
        r2_vals[j] = r2_score(ind_act[:, k], ind_pred[:, k])
    return r2_vals

def calculate_measurements(results, algorithms, base_names):
    '''For plotting results, sorting the data for each algorithm

    '''

    # Initialization (algorithm -> each metric -> each value in each trial)
    meas_list = {}
    # For mean and median
    mean_vals = {}
    median_vals = {}
    metric_names = ['rval_h', 'rval_k', 'rval_a', 'r2_h', 'r2_k', 'r2_a']
    joints = {0:'Hip', 1:'Knee', 2:'Ankle'}

    for i, name in enumerate(algorithms):
        # abbreviated algorithm name
        a_name = base_names[i]
        print(f"Processing {a_name}")
        # Initialize (children will be different metrics)
        meas_list[a_name] = {}
        mean_vals[a_name] = {}
        median_vals[a_name] = {}
        # Initializing (storing list of metric values)
        for m in metric_names:
            meas_list[a_name][m] = []
        # Extract the results for a particular algorithm
        alg_results = results[name]
        alg_actual = alg_results['actual']
        alg_pred = alg_results['pred']
        # Get necessary info
        num_trial = len(alg_actual)
        key_trial = list(alg_actual.keys())
        # Make sure there's data
        if num_trial is not 0:
            # Calculate the measurements per trial
            for t in range(num_trial):
                # Extract individual results
                trialID = key_trial[t]
                ind_act = alg_actual[trialID]
                ind_pred = alg_pred[trialID]
                # Calculate the measurements
                r_vals = calc_rval(ind_act, ind_pred, joints)
                r2_vals = calc_r2(ind_act, ind_pred, joints)
                # Log the results
                for i, m in enumerate(metric_names):
                    # If m is r-values
                    if m in metric_names[:3]:
                        meas_list[a_name][m].append(r_vals[joints[i]])
                    elif m in metric_names[3:]:
                        meas_list[a_name][m].append(r2_vals[joints[i-3]])
            # Now calculate the mean and median
            for m in metric_names:
                print("Hi")
                std_vals[a_name][m] = stats.sem(np.array(meas_list[a_name][m]))
                median_vals[a_name][m] = np.median(np.array(meas_list[a_name][m]))
        else:  # num_trial is 0
            for m in metric_names:
                meas_list[a_name][m].append(0)

    return meas_list, std_vals, median_vals, metric_names
