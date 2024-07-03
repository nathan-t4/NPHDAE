import matplotlib.pyplot as plt
import numpy as np
# Plot T vs state RMSE
plt.figure()
T = np.arange(1, 6)
data = [[524.0616455078125, 1482.11962890625, 2003.9599609375, 506.74249267578125, 1449.793701171875],
        [238.92788696289062, 689.1787719726562, 910.6741333007812, 229.63754272460938, 673.7493896484375],
        [149.02774047851562, 419.3573303222656, 570.3321533203125, 145.87619018554688, 412.1934509277344],
        [110.6522445678711, 314.995361328125, 422.86260986328125, 107.9275894165039, 309.3407897949219],
        [162.0868682861328, 343.267822265625, 651.9490356445312, 169.3740997314453, 350.176025390625]]
ts = 1500
mse = (np.sum(data, axis=1)) / (5 * 1500)
plt.plot(T, mse)
plt.title('Composite system state error')
plt.ylabel(r'$MSE$')
plt.xlabel(r'Prediction time $(T)$')
plt.xticks(T)
plt.show()

# TODO: compare with composite system simulation

# Plot T vs subsystem 1 error
errors_one = [3.7138257026672363, 4.824552297592163, 0.18629540875554085, 0.30624260753393173, 4.271171450614929]
mse_one = np.array(errors_one) / (3 * ts)

plt.figure()
plt.title('Subsystem 1 next state prediction error')
plt.plot(T, mse_one, label='Learned pH')
plt.ylabel(r'$MSE$')
plt.xlabel(r'Prediction time $(T)$')
plt.xticks(T)
plt.show()

data = [[0.42468005418777466, 1.3138715028762817, 1.7031967639923096, 0.4272685647010803, 1.3268541097640991],
        [149.02774047851562, 419.3573303222656, 570.3321533203125, 145.87619018554688, 412.1934509277344]]
mse = (np.sum(data, axis=1)) / (5 * ts)
plt.figure()
plt.scatter([3], mse[0], label='Known J')
plt.scatter([3], mse[1], label='Learned J')
plt.ylabel(r'$MSE$')
plt.xlabel(r'Prediction time $(T)$')
plt.xticks([3])
plt.legend()
plt.show()
