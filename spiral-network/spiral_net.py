import numpy as np
import matplotlib.pyplot as plt


def generate_real_data(n_samples=1000):
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples,1))
    x = t * np.cos(t)
    y = t * np.sin(t)
    spiral_data = np.concatenate((x,y), axis=1)
    spiral_data /= 10.0
    return spiral_data


def init_params(input_size, hidden_size, output_size):
    weights_1 = np.random.randn(input_size, hidden_size)*0.1
    bias_1 = np.zeros((1,hidden_size))
    weights_2 = np.random.randn(hidden_size, output_size)*0.1
    bias_2 = np.zeros((1, output_size))
    return weights_1, bias_1, weights_2, bias_2


def forward_pass_generator(latent_noise, weights_1,bias_1, weights_2, bias_2):
    hidden_pre_activation = np.dot(latent_noise, weights_1) + bias_1
    hidden_output = np.maximum(0,hidden_pre_activation)
    fake_points = np.dot(hidden_output, weights_2) + bias_2
    return fake_points, hidden_output, hidden_pre_activation

def forward_pass_discriminator(input_data, weights_1, bias_1, weights_2, bias_2):
    hidden_pre_activation = np.dot(input_data, weights_1) + bias_1
    hidden_output = np.maximum(0, hidden_pre_activation)

    output_pre_activation = np.dot(hidden_output, weights_2) + bias_2
    probability = 1 / (1+ np.exp(-output_pre_activation))

    return probability, hidden_output, hidden_pre_activation

def update_discriminator(real_data, latent_noise, params_D, params_G, n_samples, lr=0.01):
    dW1, db1, dW2, db2 = params_D
    gW1, gb1, gW2, gb2 = params_G

    fake_data, _, _ = forward_pass_generator(latent_noise, gW1, gb1, gW2, gb2)

    prob_real, hidden_real, pre_act_real = forward_pass_discriminator(real_data, dW1, db1, dW2, db2)
    prob_fake, hidden_fake, pre_act_fake = forward_pass_discriminator(fake_data, dW1, db1, dW2, db2)

    error_real = (prob_real-1) / n_samples
    error_fake = (prob_fake-0) / n_samples

    grad_W2 = np.dot(hidden_real.T, error_real) + np.dot(hidden_fake.T, error_fake)
    grad_b2 = np.sum(error_real, axis=0, keepdims=True) + np.sum(error_fake, axis=0, keepdims=True)

    error_hidden_real = np.dot(error_real, dW2.T)*(pre_act_real>0)
    error_hidden_fake = np.dot(error_fake, dW2.T)*(pre_act_fake>0)

    grad_W1 = np.dot(real_data.T, error_hidden_real) + np.dot(fake_data.T, error_hidden_fake)
    grad_b1 = np.sum(error_hidden_fake, axis=0, keepdims=True) + np.sum(error_hidden_real, axis=0, keepdims=True)

    dW1-=lr*grad_W1
    db1-=lr*grad_b1
    dW2-=lr*grad_W2
    db2-=lr*grad_b2

    return dW1, db1, dW2, db2

def update_generator(latent_noise, params_D, params_G, n_samples, lr=0.01):
    dW1, db1, dW2, db2 = params_D
    gW1, gb1, gW2, gb2 = params_G

    fake_data, gen_hidden, gen_pre_act = forward_pass_generator(latent_noise, gW1, gb1, gW2, gb2)
    prob_fake, disc_hidden, disc_pre_act = forward_pass_discriminator(fake_data, dW1, db1, dW2, db2)

    error_disc_output = (prob_fake-1.0)/n_samples

    error_disc_hidden = np.dot(error_disc_output, dW2.T) * (disc_pre_act>0)

    error_crossing_bridge = np.dot(error_disc_hidden, dW1.T)

    grad_gW2 = np.dot(gen_hidden.T, error_crossing_bridge)
    grad_b2 = np.sum(error_crossing_bridge, axis=0, keepdims=True)

    error_gen_hidden = np.dot(error_crossing_bridge, gW2.T)*(gen_pre_act>0)

    grad_gW1 = np.dot(latent_noise.T, error_gen_hidden)
    grad_b1 = np.sum(error_gen_hidden, axis=0, keepdims=True)


    gW1 -= lr * grad_gW1
    gb1 -= lr * grad_b1
    gW2 -= lr * grad_gW2
    gb2 -= lr * grad_b2

    return gW1, gb1, gW2, gb2



def visualize_progress(real_spiral, fake_spiral, iteration):
    plt.figure(figsize=(6,6))
    plt.scatter(real_spiral[:, 0], real_spiral[:, 1], color='blue', alpha=0.3, label='Real')
    plt.scatter(fake_spiral[:, 0], fake_spiral[:, 1], color='red', alpha=0.5, label='Artist')
    plt.title(f"Iteration {iteration}")
    plt.xlim(-1.5, 1.5); plt.ylim(-1.5, 1.5)
    plt.legend(); plt.show()

# --- Training Setup ---
gen_params = init_params(2, 128, 2)
disc_params = init_params(2, 128, 1)

n_samples = 1000
iterations = 20001
learning_rate = 0.1
real_spiral = generate_real_data(n_samples)


for i in range(10001):
    noise = np.random.randn(n_samples, 2)
    disc_params = update_discriminator(real_spiral, noise, disc_params, gen_params, n_samples, lr=learning_rate)
    gen_params = update_generator(noise, disc_params, gen_params, n_samples, lr=learning_rate)
    
    if i % 2000 == 0:
        fake_data, _, _ = forward_pass_generator(noise, *gen_params)
        visualize_progress(real_spiral, fake_data, i)
        print(f"Iteration {i} complete.")