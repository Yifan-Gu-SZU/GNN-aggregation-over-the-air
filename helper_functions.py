import numpy as np
import scipy as sp
from scipy import special

def layout_generate(general_para):
    N = general_para.n_links
    # first, generate transmitters' coordinates
    tx_xs = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])
    tx_ys = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])
    while(True): # loop until a valid layout generated
        # generate rx one by one rather than N together to ensure checking validity one by one
        rx_xs = []; rx_ys = []
        for i in range(N):
            got_valid_rx = False
            while(not got_valid_rx):
                pair_dist = np.random.uniform(low=general_para.shortest_directLink_length, high=general_para.longest_directLink_length)
                pair_angles = np.random.uniform(low=0, high=np.pi*2)
                rx_x = tx_xs[i] + pair_dist * np.cos(pair_angles)
                rx_y = tx_ys[i] + pair_dist * np.sin(pair_angles)
                if(0<=rx_x<=general_para.field_length and 0<=rx_y<=general_para.field_length):
                    got_valid_rx = True
            rx_xs.append(rx_x); rx_ys.append(rx_y)
        # For now, assuming equal weights and equal power, so not generating them
        layout = np.concatenate((tx_xs, tx_ys, rx_xs, rx_ys), axis=1)
        distances = np.zeros([N, N])
        # compute distance between every possible Tx/Rx pair
        for rx_index in range(N):
            for tx_index in range(N):
                tx_coor = layout[tx_index][0:2]
                rx_coor = layout[rx_index][2:4]
                # according to paper notation convention, Hij is from jth transmitter to ith receiver
                distances[rx_index][tx_index] = np.linalg.norm(tx_coor - rx_coor)
        # Check whether a tx-rx link (potentially cross-link) is too close
        if(np.min(distances)>general_para.shortest_crossLink_length):
                 break
    return layout, distances

def compute_SINRs(general_para, allocs, directlink_channel_losses, crosslink_channel_losses):
    assert np.shape(directlink_channel_losses) == np.shape(allocs), \
        "Mismatch shapes: {} VS {}".format(np.shape(directlink_channel_losses), np.shape(allocs))
    SINRs_numerators = allocs * directlink_channel_losses 
    SINRs_denominators = np.squeeze(np.matmul(crosslink_channel_losses, np.expand_dims(allocs, axis=-1))) + general_para.output_noise_power / general_para.tx_power  # layouts X N
    SINRs = SINRs_numerators / SINRs_denominators 
    return SINRs

def compute_rates(general_para, allocs, directlink_channel_losses, crosslink_channel_losses):
    SINRs = compute_SINRs(general_para, allocs, directlink_channel_losses, crosslink_channel_losses)
    rates = np.log2(1 + SINRs)
    return rates   

def get_directLink_channel_losses(channel_losses):
    return np.diagonal(channel_losses, axis1=1, axis2=2) 

def get_crossLink_channel_losses(channel_losses):
    N = np.shape(channel_losses)[-1]
    return channel_losses * ((np.identity(N) < 1).astype(float))

#r is the channel correlation coefficient
def add_fast_fading_sequence(timesteps, train_path_losses):
    n = np.shape(train_path_losses)
    n_links = np.multiply(n[1],n[2])
    channel_losses_sequence = np.zeros((n[0],timesteps,n[1],n[2]))
    for i in range(n[0]):
        r = np.random.rand()
        alpha = np.resize(train_path_losses[i,:,:],n_links)
        noise_var = np.multiply(alpha,1-np.power(r,2))
        # channel coefficient matrix
        sims_real = np.zeros((timesteps,n_links))
        sims_imag = np.zeros((timesteps,n_links))
    # generate the channel coefficients for consecutive frames
        sims_real[0,:] = np.random.normal(loc = 0, scale = np.sqrt(alpha))
        sims_imag[0,:] = np.random.normal(loc = 0, scale = np.sqrt(alpha))
        for j in range(timesteps-1):
            sims_real[j+1,:] = np.multiply(r,sims_real[j,:]) + np.random.normal(loc = 0, scale = np.sqrt(noise_var))
            sims_imag[j+1,:] = np.multiply(r,sims_imag[j,:]) + np.random.normal(loc = 0, scale = np.sqrt(noise_var))
        layout_channel_losses_sequence = (np.power(sims_real, 2) + np.power(sims_imag, 2))/2
        channel_losses_sequence[i,:,:,:] = np.resize(layout_channel_losses_sequence,(timesteps,n[1],n[2]))
    return channel_losses_sequence

def batch_WMMSE(p_int, alpha, H, Pmax, var_noise):
    N = p_int.shape[0]
    K = p_int.shape[1]
    vnew = 0
    b = np.sqrt(p_int)
    f = np.zeros((N,K,1) )
    w = np.zeros( (N,K,1) )
    

    mask = np.eye(K)
    rx_power = np.multiply(H, b)
    rx_power_s = np.square(rx_power)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)
    
    interference = np.sum(rx_power_s, 2) + var_noise
    f = np.divide(valid_rx_power,interference)
    w = 1/(1-np.multiply(f,valid_rx_power))
    #vnew = np.sum(np.log2(w),1)
    
    for ii in range(100):
        fp = np.expand_dims(f,1)
        rx_power = np.multiply(H.transpose(0,2,1), fp)
        valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)
        bup = np.multiply(alpha,np.multiply(w,valid_rx_power))
        rx_power_s = np.square(rx_power)
        wp = np.expand_dims(w,1)
        alphap = np.expand_dims(alpha,1)
        bdown = np.sum(np.multiply(alphap,np.multiply(rx_power_s,wp)),2)
        btmp = bup/bdown
        b = np.minimum(btmp, np.ones((N,K) )*np.sqrt(Pmax)) + np.maximum(btmp, np.zeros((N,K) )) - btmp
        
        bp = np.expand_dims(b,1)
        rx_power = np.multiply(H, bp)
        rx_power_s = np.square(rx_power)
        valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)
        interference = np.sum(rx_power_s, 2) + var_noise
        f = np.divide(valid_rx_power,interference)
        w = 1/(1-np.multiply(f,valid_rx_power))
    p_opt = np.square(b)
    return p_opt