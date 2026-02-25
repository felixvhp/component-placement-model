import matplotlib.pyplot as plt
import os
import numpy as np

def plot_training_loss(loss, overlap_loss, volume_loss, 
                       compactness_loss, ldv_loss, dpp_aspect_ratios_loss,
                       infonce_loss, loss_weights_history):

    loss_weights_history = np.array(loss_weights_history)
    λ_overlap = loss_weights_history[:, 0]
    λ_volume = loss_weights_history[:, 1]
    λ_compactness = loss_weights_history[:, 2]
    λ_ldv = loss_weights_history[:, 3]
    λ_dpp_aspect_ratios = loss_weights_history[:, 4]
    λ_infonce = loss_weights_history[:, 5]

    loss = np.array(loss)
    overlap_loss = np.array(overlap_loss)
    volume_loss = np.array(volume_loss)
    compactness_loss = np.array(compactness_loss)
    ldv_loss = np.array(ldv_loss)
    dpp_aspect_ratios_loss = np.array(dpp_aspect_ratios_loss)
    mi_loss = np.array(infonce_loss)
    
    eps = 1e-6        # to avoid log(0) --> better axis lims
    window_size = 50
    epochs = np.arange(len(loss))

    overlap_loss[:] += eps
    volume_loss[:] += eps
    compactness_loss[:] += eps
    ldv_loss[:] += eps  
    dpp_aspect_ratios_loss[:] += eps
    mi_loss[:] += eps

    def moving_average(data):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    colors = ['black', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))

    ax[0, 0].plot(moving_average(epochs), moving_average(loss), label='Total Loss', color = 'black')
    ax[0, 0].plot(loss, color = 'black', alpha = 0.4)

    ax[0, 0].plot(moving_average(epochs), moving_average(overlap_loss), label='Overlap Loss', color = colors[1])
    ax[0, 0].plot(overlap_loss, color = colors[1], alpha = 0.4)
    ax[0, 0].plot(moving_average(epochs), moving_average(volume_loss), label='Volume Loss', color = colors[2])
    ax[0, 0].plot(volume_loss, color = colors[2], alpha = 0.4)    
    ax[0, 0].plot(moving_average(epochs), moving_average(compactness_loss), label='Compactness Loss', color = colors[3])
    ax[0, 0].plot(compactness_loss, color = colors[3], alpha = 0.4)
    ax[0, 0].plot(moving_average(epochs), moving_average(dpp_aspect_ratios_loss), label='DPP Aspect Ratios Loss', color = colors[5])
    ax[0, 0].plot(dpp_aspect_ratios_loss, color = colors[5], alpha = 0.4)
    ax[0, 0].plot(moving_average(epochs), moving_average(mi_loss), label='MI Loss', color = colors[6])
    ax[0, 0].plot(mi_loss, color = colors[6], alpha = 0.4)
    ax[0, 0].set_xlabel('Iteration')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].legend(frameon=False)


    ax[0, 1].plot(moving_average(epochs), moving_average(loss), label='Total Loss', color = 'black')
    ax[0, 1].plot(loss, color = 'black', alpha = 0.4)
    
    ax[0, 1].plot(moving_average(epochs), moving_average(overlap_loss), label='Overlap Loss', color = colors[1])
    ax[0, 1].plot(overlap_loss, color = colors[1], alpha = 0.4)
    ax[0, 1].plot(moving_average(epochs), moving_average(volume_loss), label='Volume Loss', color = colors[2])
    ax[0, 1].plot(volume_loss, color = colors[2], alpha = 0.4)
    ax[0, 1].plot(moving_average(epochs), moving_average(compactness_loss), label='Compactness Loss', color = colors[3])
    ax[0, 1].plot(compactness_loss, color = colors[3], alpha = 0.4)
    ax[0, 1].plot(moving_average(epochs), moving_average(dpp_aspect_ratios_loss), label='DPP Aspect Ratios Loss', color = colors[5])
    ax[0, 1].plot(dpp_aspect_ratios_loss, color = colors[5], alpha = 0.4)
    ax[0, 1].plot(moving_average(epochs), moving_average(ldv_loss), label='LDV Loss', color = colors[4])
    ax[0, 1].plot(ldv_loss, color = colors[4], alpha = 0.4)
    ax[0, 1].plot(moving_average(epochs), moving_average(mi_loss), label='MI Loss', color = colors[6])
    ax[0, 1].plot(mi_loss, color = colors[6], alpha = 0.4)

    ax[0, 1].set_xlabel('Iteration')
    ax[0, 1].set_ylabel('Loss')
    ax[0, 1].set_yscale('log')
    ax[0, 1].legend(frameon=False)

    ax[1, 0].plot(moving_average(epochs), moving_average(ldv_loss), label='LDV Loss', color = colors[4])
    ax[1, 0].plot(ldv_loss, color = colors[4], alpha = 0.4)
    ax[1, 0].plot(moving_average(epochs), moving_average(mi_loss), label='MI Loss', color = colors[6])
    ax[1, 0].plot(mi_loss, color = colors[6], alpha = 0.4)
    ax[1, 0].plot(moving_average(epochs), moving_average(dpp_aspect_ratios_loss), label='DPP Aspect Ratios Loss', color = colors[5])
    ax[1, 0].plot(dpp_aspect_ratios_loss, color = colors[5], alpha = 0.4)
    ax[1, 0].set_xlabel('Iteration')
    ax[1, 0].set_ylabel('Loss')
    ax[1, 0].legend(frameon=False)

    # Create weighted stacked plot
    ma_epochs = moving_average(epochs)
    weighted_overlap = moving_average(overlap_loss * λ_overlap) / moving_average(loss)
    weighted_volume = moving_average(volume_loss * λ_volume) / moving_average(loss)
    weighted_compactness = moving_average(compactness_loss * λ_compactness) / moving_average(loss)
    weighted_ldv = moving_average(ldv_loss * λ_ldv) / moving_average(loss)
    weighted_dpp_aspect_ratios = moving_average(dpp_aspect_ratios_loss * λ_dpp_aspect_ratios) / moving_average(loss)
    weighted_mi = moving_average(mi_loss * λ_infonce) / moving_average(loss)

    ax[1, 1].stackplot(ma_epochs,   [weighted_overlap, weighted_volume, weighted_compactness, weighted_ldv, weighted_dpp_aspect_ratios, weighted_mi],
                      labels=['Weighted Overlap Loss', 'Weighted Volume Loss', 'Weighted Compactness Loss', 'Weighted LDV Loss', 'Weighted DPP Aspect Ratios Loss', 'Weighted MI Loss'],
                      colors=[colors[1], colors[2], colors[3], colors[4], colors[5], colors[6]], alpha=0.5)
    ax[1, 1].set_xlabel('Iteration')
    ax[1, 1].set_ylabel('Weighted Loss')
    ax[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'training_loss.png'))
    plt.close()